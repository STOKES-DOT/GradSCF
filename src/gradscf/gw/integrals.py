"""Selected molecular GW vertices; AO2MO algorithms remain in gradscf.ao2mo."""
import jax.numpy as jnp
import jax

from .._memory import DEFAULT_MAX_MEMORY_MB, array_storage_bytes
from ..ao2mo.df import transform_df_factors


def auxiliary_factors(mean_field, auxiliary_basis, *, max_memory):
    """Independent, fixed-geometry libcint RI; never rerun/change the SCF.

    Native 2c/3c preparation is on CPU. The present integral supplier returns
    full packed 3c data; bound its known simultaneous owners before invoking
    it. This is not an out-of-core RI provider or a full RI geometry AD map.
    """
    from .. import integrals
    from .._memory import tree_storage_bytes, workspace_bytes, array_device
    from ..integrals.molecular.density_fitting import unpack_factors
    resident = tree_storage_bytes(getattr(mean_field, "_scf_inputs", None),
        getattr(mean_field, "scf_result", None), getattr(mean_field, "reference", None))
    device = array_device(mean_field.mo_energy)
    with jax.default_device(jax.devices("cpu")[0]):
        topology, parameters = integrals.prepare_basis(mean_field._spec(),
            mean_field.mol.basis, cart=mean_field.mol.cart)
        auxiliary, ap = integrals.prepare_basis(mean_field._spec(), auxiliary_basis,
                                               cart=mean_field.mol.cart)
        n, p = topology.nao, auxiliary.nao
        if device is not None and device.platform != "cpu":
            destination = workspace_bytes(max_memory, resident_bytes=resident, device=device)
            if 8*p*n*n > destination:
                raise ValueError("RI factors exceed the destination-device memory budget before transfer")
        minimum = 8*(5*p*n*(n+1)//2+2*p*n*n+8*p*p)
        available = workspace_bytes(max_memory, resident_bytes=resident, device=jax.devices("cpu")[0])
        if minimum > available:
            raise ValueError("Known RI preparation workspace exceeds max_memory; full packed 3c data cannot fit")
        packed = integrals.make_auxiliary_plan(topology, auxiliary).factors(parameters, ap)
        factors = unpack_factors(packed, n)
    return factors if device is None else jax.device_put(factors, device)


def starting_potential(mean_field, density, fock, hcore, *, max_memory):
    """F-h-J from the ACTUAL SCF representation, independently of GW fitting."""
    from ..integrals.molecular.jk import build_j_from_df, build_jk_from_packed
    source = getattr(mean_field, "_scf_inputs", None)
    if source is None:
        source = mean_field.reference
    factors = getattr(source, "df_factors", None)
    direct = getattr(source, "direct_basis", None)
    pair = getattr(source, "eri_pair_matrix", None)
    eri = getattr(source, "eri", None)
    if eri is None:
        eri = getattr(source, "rep_tensor", None)
    if factors is not None:
        coulomb = build_j_from_df(factors, density)
    elif direct is not None:
        from ..integrals.backends.native.jk import _selected_jk
        coulomb, _ = _selected_jk(direct, density, with_k=False,
            screening_threshold=getattr(mean_field, "direct_scf_tol", 0.),
            workers=getattr(mean_field, "direct_scf_workers", 1))
    elif pair is not None:
        coulomb, _ = build_jk_from_packed(pair, density, with_k=False, max_memory=max_memory)
    elif eri is not None and eri.ndim in (1, 2):
        coulomb, _ = build_jk_from_packed(eri, density, with_k=False, max_memory=max_memory)
    elif eri is not None and eri.ndim == 4:
        coulomb = jnp.einsum("pqrs,rs->pq", eri, density, precision="highest")
    else:
        raise RuntimeError("The SCF reference does not expose its actual Coulomb action")
    return jnp.asarray(fock)-jnp.asarray(hcore)-coulomb


def prepare_selected_factors(factors, coefficients, *, nocc, orbs, g_orbitals,
                             screening_occupied, screening_virtual,
                             max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Prepare distinct ordered vertices without the full Qmn array.

    Windows are validated original-MO indices from the driver. Raw AO factors
    need not be symmetric, so the two vertices are transformed independently.
    Each AO2MO call reserves already prepared outputs under the same budget.
    """
    coefficients = jnp.asarray(coefficients)
    output = {}
    reservation = reserved_bytes+array_storage_bytes(coefficients)
    def transform(left, right):
        c_left = coefficients[:, jnp.asarray(left, dtype=jnp.int32)]
        c_right = coefficients[:, jnp.asarray(right, dtype=jnp.int32)]
        return transform_df_factors(factors, c_left, c_right,
            max_memory=max_memory,
            reserved_bytes=reservation+array_storage_bytes(*output.values()))
    output["b_ov"] = transform(screening_occupied, screening_virtual)
    output["b_pm"] = transform(orbs, g_orbitals)
    output["b_mp"] = transform(g_orbitals, orbs).transpose(0, 2, 1)
    # Bare exchange uses ALL reference occupied orbitals, independently of
    # the G/W correlation windows. Its output is only the requested diagonal.
    occupied = tuple(range(nocc))
    left = transform(orbs, occupied)
    right = transform_df_factors(factors,
        coefficients[:, :nocc], coefficients[:, jnp.asarray(orbs, dtype=jnp.int32)],
        max_memory=max_memory, reserved_bytes=reservation+
        array_storage_bytes(*output.values(), left)).transpose(0, 2, 1)
    output["exchange"] = -jnp.einsum("Qpi,Qpi->p", left, right, precision="highest")
    return output
