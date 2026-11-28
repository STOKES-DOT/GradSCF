"""Closed-core folding and selected two-electron integrals in a common MO frame."""

from math import prod
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from gradscf._memory import (
    DEFAULT_MAX_MEMORY_MB, array_device, array_storage_bytes, resolve_memory_mb,
    tree_storage_bytes, validate_reserved_bytes, workspace_bytes,
)
from gradscf.ao2mo.blocks import transform_eri_blocks
from gradscf.ao2mo.reference import transform_integrals, validate_integrals
from gradscf.integrals.molecular.jk import (
    build_jk_from_df, build_jk_from_packed, exchange_matrix,
)


def orbital_selection(norb, core=0, active=None):
    if isinstance(core, Integral) and not isinstance(core, bool):
        if not 0 <= core <= norb:
            raise ValueError("core count must be between 0 and norb")
        core = tuple(range(core))
    else:
        try:
            core = tuple(core)
        except TypeError as error:
            raise ValueError("core must be a count or orbital index sequence") from error
    try:
        active = None if active is None else tuple(active)
    except TypeError as error:
        raise ValueError("active must be an orbital index sequence") from error
    for name, indices in (("core", core), ("active", () if active is None else active)):
        invalid = any(not isinstance(i, Integral) or isinstance(i, bool) or not 0 <= i < norb
                      for i in indices)
        if invalid or len(set(indices)) != len(indices):
            raise ValueError(f"{name} indices must be distinct integers in [0,norb)")
    active = tuple(p for p in range(norb) if p not in core) if active is None else tuple(active)
    if set(core) & set(active):
        raise ValueError("Core and active orbital sets must be disjoint")
    return tuple(map(int, core)), tuple(map(int, active))


def read_ao_representation(inputs, coefficients, *, max_memory=DEFAULT_MAX_MEMORY_MB,
                           reserved_bytes=0, owner=None, allow_legacy=True):
    """Prefer an existing SCF representation without transforming any orbitals."""
    from gradscf.integrals.backends.native.jk import (
        NativeDirectBasis, ProjectedNativeDirectBasis,
    )

    factors = getattr(inputs, "df_factors", None)
    if factors is not None:
        return {"df_factors": factors}
    eri = getattr(inputs, "eri", None)
    if eri is not None and np.size(eri):
        return {"eri" if np.ndim(eri) == 4 else "eri_pair_matrix": eri}
    pair = getattr(inputs, "eri_pair_matrix", None)
    if pair is not None and np.size(pair):
        return {"eri_pair_matrix": pair}
    direct = getattr(inputs, "direct_basis", None)
    if isinstance(direct, (NativeDirectBasis, ProjectedNativeDirectBasis)):
        return {"direct_basis": direct}
    if allow_legacy and hasattr(inputs, "response_eri_pair_matrix"):
        nao = coefficients.shape[0]
        npair = nao * (nao + 1) // 2
        itemsize = max(8, jnp.result_type(coefficients, inputs.hcore).itemsize)
        resident = reserved_bytes + tree_storage_bytes(
            inputs if owner is None else owner, coefficients,
        )
        available = workspace_bytes(
            max_memory, resident_bytes=resident, device=array_device(coefficients),
        )
        # Legacy packed construction retains an accumulator, a chunk and
        # their sum, including one sentinel AO-pair row.
        if 3 * (npair + 1) ** 2 * itemsize > available:
            raise ValueError("Legacy packed AO ERI construction buffers exceed the working-memory budget")
        pair = inputs.response_eri_pair_matrix()
    if pair is None:
        raise ValueError("SCF source supplies no transformable two-electron integrals")
    return {"eri_pair_matrix": pair}


def _check_native_active_memory(hcore, coefficients, constant, basis, ncore, nactive,
                                max_memory, reserved_bytes=0, retained_bytes=0):
    """Check known CAS/core arrays before native J/K or shell evaluation.

    The block transformer additionally checks its minimum shell tile. These
    forward-array estimates do not bound complete solver/AD process residency.
    """
    from gradscf.integrals.backends.native.jk import (
        NativeDirectBasis, ProjectedNativeDirectBasis, native_jk_workspace_bytes,
    )
    nao = coefficients.shape[0]
    required = 8 * (3*nactive**4 + 3*nactive**2 + nao*(ncore+nactive)) + retained_bytes
    primitive = basis
    while isinstance(primitive,ProjectedNativeDirectBasis):
        transform = primitive.transform
        # Promotion/projection buffers, plus primitive density/J/K versions
        # retained by the closed-core action, use the native float64 ABI.
        required += 8*transform.shape[0]*(transform.shape[1]+ncore+nactive)
        if ncore:
            required += 8*4*transform.shape[0]**2
        primitive = primitive.primitive_basis
    if not isinstance(primitive,NativeDirectBasis):
        raise TypeError("direct_basis must be a native direct integral basis")
    if ncore:
        required += 8*8*nao**2 + native_jk_workspace_bytes(primitive,1,0.)
    arrays = (hcore,coefficients,constant,*jax.tree.leaves(basis))
    available = workspace_bytes(
        max_memory, resident_bytes=reserved_bytes+array_storage_bytes(*arrays),
        device=array_device(*arrays),
    )
    if required > available:
        raise ValueError("Native active/core arrays exceed the working-memory budget")


def _active_inputs(hcore, coefficients, core, active, ecore, max_memory,
                   reserved_bytes, **sources):
    max_memory = resolve_memory_mb(max_memory)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    hcore,coefficients,constant = map(jnp.asarray,(hcore,coefficients,ecore))
    if coefficients.ndim != 2 or hcore.shape != (coefficients.shape[0],)*2:
        raise ValueError("Inconsistent AO Hamiltonian and MO coefficient dimensions")
    if jnp.iscomplexobj(hcore) or jnp.iscomplexobj(coefficients):
        raise NotImplementedError("Active-space transformation requires real orbitals and integrals")
    if constant.shape != () or jnp.iscomplexobj(constant):
        raise ValueError("ecore must be a real scalar")
    core,active = orbital_selection(coefficients.shape[1],core,active)
    representation = {key: value for key, value in sources.items() if value is not None}
    if len(representation) != 1:
        raise ValueError("Supply exactly one of eri, eri_pair_matrix, df_factors, direct_basis")
    return hcore, coefficients, constant, core, active, max_memory, reserved_bytes, representation


def core_potential(density, representation, *, symmetric=True):
    """Evaluate J[D] - K[D]/2 for a closed-core or transition AO density.

    For transition densities the exchange is generally nonsymmetric. The
    density convention is two times ket core coefficients times bra core
    coefficients transposed, with the ket AO index first.
    """
    direct = representation.get("direct_basis")
    factors = representation.get("df_factors")
    packed = representation.get("eri_pair_matrix")
    if direct is not None:
        coulomb, exchange = direct.get_jk(density, screening_threshold=0., workers=1)
    elif factors is not None:
        coulomb, exchange = build_jk_from_df(factors, density, symmetrize=symmetric)
    elif packed is not None:
        coulomb, exchange = build_jk_from_packed(packed, density)
    else:
        eri = representation.get("eri")
        if eri is None:
            raise ValueError("An AO integral representation is required for core folding")
        coulomb = jnp.einsum("pqrs,rs->pq", eri, density, precision="highest")
        exchange = exchange_matrix(eri, density)
    return coulomb - .5 * exchange


def transform_active(hcore, coefficients, *, core=0, active=None, ecore=0.,
                          max_memory=DEFAULT_MAX_MEMORY_MB, eri=None,
                          eri_pair_matrix=None, df_factors=None, direct_basis=None,
                          reserved_bytes=0, _transform=transform_integrals):
    """Prepare a fixed real spatial frame without eager SCF validation.

    Return (h_active, eri_active, Ecore), preserving supplied active order.
    Coefficients, one-electron/nuclear inputs and the native provider remain
    live JAX inputs; this does not differentiate an SCF source or choose MOs.
    Native coordinates retain the shared direct operators' order-two AD
    boundary. Only the necessary nactive**4 MO ERI output is produced.
    """
    (hcore, coefficients, constant, core, active, max_memory, reserved_bytes,
     representation) = _active_inputs(
        hcore, coefficients, core, active, ecore, max_memory, reserved_bytes,
        eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors,
        direct_basis=direct_basis,
    )
    if direct_basis is not None:
        _check_native_active_memory(
            hcore, coefficients, constant, direct_basis, len(core), len(active),
            max_memory, reserved_bytes,
        )
    ca = coefficients[:, jnp.asarray(active, dtype=jnp.int32)]

    def transform(effective, *retained):
        reserved = reserved_bytes+tree_storage_bytes(hcore, coefficients, ecore, constant, *retained,
                                                     exclude=(effective, ca))
        return _transform(effective, ca, max_memory=max_memory,
                                    reserved_bytes=reserved, **representation)

    if direct_basis is not None:
        # The block planner's minimum shell-tile check precedes core J/K.
        h,g = transform(hcore)
    effective = hcore
    if core:
        cc = coefficients[:,jnp.asarray(core,dtype=jnp.int32)]
        density = 2 * cc @ cc.T
        potential = core_potential(density, representation)
        constant += jnp.sum(density * (hcore + .5 * potential).T)
        effective = hcore+potential
        if direct_basis is not None:
            h = h + ca.T@potential@ca
    # Core folding stays in AO space. Only nactive^4 MO ERIs are produced,
    # even if many core orbitals are frozen.
    if direct_basis is None:
        h,g = transform(effective, cc, density, potential) if core else transform(effective)
    return h,g,constant


def transform_active_blocks(
    hcore, coefficients, *, blocks, core=0, active=None, ecore=0.,
    max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0, eri=None,
    eri_pair_matrix=None, df_factors=None, direct_basis=None,
):
    """Return (h_A, g_AAAA, E_core, extra ERIs) for one common MO frame.

    Each static four-character request names the orbitals on each leg: "p"
    and "q" mean all MO columns, "a" active, "c" doubly occupied core, and
    "v" external. Block order and the order within each space are preserved.
    The native backend shares shell traversal across all requested blocks.
    Coefficients, representations and nuclear terms remain live JAX inputs.
    """
    (hcore, coefficients, constant, core, active, max_memory, reserved_bytes,
     representation) = _active_inputs(
        hcore, coefficients, core, active, ecore, max_memory, reserved_bytes,
        eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors,
        direct_basis=direct_basis,
    )
    blocks = tuple(blocks)
    if any(not isinstance(block, str) or len(block) != 4 or set(block) - set("pqacv")
           for block in blocks):
        raise ValueError('Blocks must name four static MO spaces using "pqacv"')
    labels = tuple(dict.fromkeys("a" + "".join(blocks).replace("q", "p")))
    external = tuple(i for i in range(coefficients.shape[1]) if i not in core and i not in active)
    indices = {"a": active, "c": core, "v": external}
    requests = tuple(tuple(labels.index(i) for i in block.replace("q", "p"))
                     for block in ("aaaa", *blocks))
    if direct_basis is not None:
        widths = tuple(coefficients.shape[1] if label == "p" else len(indices[label])
                       for label in labels)
        retained = 8 * sum(prod(widths[i] for i in request)
                           for request in set(requests[1:]) - {requests[0]})
        _check_native_active_memory(
            hcore, coefficients, constant, direct_basis, len(core), len(active),
            max_memory, reserved_bytes, retained_bytes=retained,
        )
    spaces = tuple(coefficients if label == "p" else
                   coefficients[:, jnp.asarray(indices[label], dtype=jnp.int32)]
                   for label in labels)
    ca = spaces[0]
    resident = tree_storage_bytes(hcore, coefficients, constant, exclude=spaces)
    one_body = jnp.result_type(hcore, coefficients).itemsize * (ca.size + ca.shape[1] ** 2)
    g, *extra = transform_eri_blocks(
        spaces, requests, max_memory=max_memory,
        reserved_bytes=reserved_bytes + resident + one_body, **representation,
    )
    effective = hcore
    if core:
        cc = spaces[labels.index("c")] if "c" in labels else coefficients[:, jnp.asarray(core)]
        density = 2 * cc @ cc.T
        potential = core_potential(density, representation)
        constant += jnp.sum(density * (hcore + .5 * potential).T)
        effective = hcore + potential
    h, g = validate_integrals(ca.T @ effective @ ca, g)
    return h, g, constant, tuple(extra)
