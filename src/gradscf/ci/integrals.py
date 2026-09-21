"""Real MO integral preparation, separated from the differentiable CI solve."""
import jax.numpy as jnp
import numpy as np

from ..integrals.layouts import _metadata_arrays, _mo_pair_products
from .hamiltonian import validate_integrals
from .types import CIReference


def transform_integrals(hcore, mo_coeff, *, eri=None, eri_pair_matrix=None, df_factors=None):
    """Transform full, s4 pair, or density-fitted AO integrals using JAX.

    Exactly one ERI representation is required. The resulting full MO ERI tensor
    uses O(nmo**4) memory; this is a small-system reference implementation.
    """
    c, h = jnp.asarray(mo_coeff), jnp.asarray(hcore)
    if jnp.iscomplexobj(c) or jnp.iscomplexobj(h):
        raise NotImplementedError("CI currently requires real orbitals and integrals")
    if c.ndim != 2 or h.shape != (c.shape[0], c.shape[0]):
        raise ValueError("Inconsistent AO Hamiltonian and MO coefficient dimensions")
    if sum(value is not None for value in (eri, eri_pair_matrix, df_factors)) != 1:
        raise ValueError("Supply exactly one of eri, eri_pair_matrix, df_factors")
    nao = c.shape[0]
    if eri is not None:
        eri = jnp.asarray(eri)
        if eri.shape != (nao,) * 4:
            raise ValueError("AO eri must have shape (nao, nao, nao, nao)")
        g = jnp.einsum("pqrs,pP,qQ,rR,sS->PQRS", eri, c, c, c, c,
                       precision="highest", optimize="optimal")
    elif eri_pair_matrix is not None:
        pair = jnp.asarray(eri_pair_matrix)
        npair = nao * (nao + 1) // 2
        if pair.shape != (npair, npair):
            raise ValueError("eri_pair_matrix must be a square s4 AO pair matrix")
        rows, cols, _, _ = _metadata_arrays(nao, c.dtype)
        products = _mo_pair_products(c, c, rows, cols)
        g = jnp.einsum("pqP,PQ,rsQ->pqrs", products, pair, products, precision="highest")
    else:
        factors = jnp.asarray(df_factors)
        if factors.ndim != 3 or factors.shape[1:] != (nao, nao):
            raise ValueError("df_factors must have shape (naux, nao, nao)")
        b = jnp.einsum("Lpq,pP,qQ->LPQ", factors, c, c, precision="highest")
        g = jnp.einsum("Lpq,Lrs->pqrs", b, b, precision="highest")
    return validate_integrals(c.T @ h @ c, g)


def reference_from_source(source):
    """Prepare explicit MO inputs or a converged GradSCF closed-shell HF facade.

    This adapter is eager. Use solve_ci/solve_cis for transformed JAX functions.
    It never calls an external electronic-structure program.
    """
    if isinstance(source, CIReference):
        h, g = validate_integrals(source.h1, source.eri)
        if not isinstance(source.nocc, int) or not 0 <= source.nocc <= h.shape[0]:
            raise ValueError("Invalid restricted nocc")
        return CIReference(h, g, source.nocc, source.nuclear_repulsion, source.mo_energy)
    from ..scf.facade import RKS

    if not isinstance(source, RKS) or str(source.xc).strip().lower() != "hf":
        raise NotImplementedError("CI requires a GradSCF closed-shell HF reference (RKS with xc='hf') or CIReference")
    if not source.converged or source.scf_result is None or source._scf_inputs is None:
        raise RuntimeError("Run and converge the HF SCF before CI")
    # Detect stale molecular parameters using the facade's existing guard.
    if source._cached_scf_key != source._scf_signature():
        raise RuntimeError("SCF inputs changed; run SCF again before CI")
    inputs = source._scf_inputs
    nocc = inputs.nelectron // 2
    c = source.mo_coeff
    if c.ndim != 2 or inputs.nelectron % 2:
        raise NotImplementedError("Only real closed-shell HF references are supported")
    expected = np.zeros(c.shape[1])
    expected[:nocc] = 2
    if not np.allclose(np.asarray(source.mo_occ), expected, atol=1e-10, rtol=0):
        raise ValueError("CI requires integer occupations, occupied orbitals first")
    if inputs.df_factors is not None:
        kwargs = {"df_factors": inputs.df_factors}
    elif inputs.eri is not None:
        kwargs = {"eri": inputs.eri}
    else:
        pair = inputs.response_eri_pair_matrix()
        if pair is None:
            raise ValueError("SCF reference does not supply CI two-electron integrals")
        kwargs = {"eri_pair_matrix": pair}
    h, g = transform_integrals(inputs.hcore, c, **kwargs)
    return CIReference(h, g, nocc, inputs.nuclear_repulsion, source.mo_energy)
