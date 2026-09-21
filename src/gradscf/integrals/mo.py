"""Shared real MO integrals and frozen-orbital selection for post-HF methods."""

from numbers import Integral
import jax.numpy as jnp
from .layouts import _metadata_arrays, _mo_pair_products


def validate_integrals(h1, eri, nmo=None):
    h1, eri = jnp.asarray(h1), jnp.asarray(eri)
    if jnp.iscomplexobj(h1) or jnp.iscomplexobj(eri):
        raise NotImplementedError(
            "MO currently requires real integrals and real orbitals"
        )
    if not jnp.issubdtype(h1.dtype, jnp.floating) or not jnp.issubdtype(
        eri.dtype, jnp.floating
    ):
        raise ValueError("MO integrals must have floating-point dtype")
    if h1.ndim != 2 or h1.shape[0] != h1.shape[1]:
        raise ValueError("h1 must be square")
    if nmo is not None and h1.shape != (nmo, nmo):
        raise ValueError("MO integral dimensions do not match the MO space")
    if eri.shape != h1.shape * 2:
        raise ValueError(
            "eri must have shape (nmo, nmo, nmo, nmo), in chemists' notation"
        )
    return h1, eri


def transform_integrals(
    hcore, mo_coeff, *, eri=None, eri_pair_matrix=None, df_factors=None
):
    """Transform full, s4 pair, or density-fitted AO integrals using JAX.

    Exactly one ERI representation is required. The resulting full MO ERI tensor
    uses O(nmo**4) memory; this is a small-system reference implementation.
    """
    c, h = jnp.asarray(mo_coeff), jnp.asarray(hcore)
    if jnp.iscomplexobj(c) or jnp.iscomplexobj(h):
        raise NotImplementedError(
            "Post-HF transformation currently requires real orbitals and integrals"
        )
    if c.ndim != 2 or h.shape != (c.shape[0], c.shape[0]):
        raise ValueError("Inconsistent AO Hamiltonian and MO coefficient dimensions")
    g = _transform_eri(c, c, eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors)
    return validate_integrals(c.T @ h @ c, g)


def _transform_eri(c, d, *, eri=None, eri_pair_matrix=None, df_factors=None):
    if sum(value is not None for value in (eri, eri_pair_matrix, df_factors)) != 1:
        raise ValueError("Supply exactly one of eri, eri_pair_matrix, df_factors")
    nao = c.shape[0]
    if eri is not None:
        eri = jnp.asarray(eri)
        if eri.shape != (nao,) * 4:
            raise ValueError("AO eri must have shape (nao, nao, nao, nao)")
        g = jnp.einsum(
            "pqrs,pP,qQ,rR,sS->PQRS",
            eri,
            c,
            c,
            d,
            d,
            precision="highest",
            optimize="optimal",
        )
    elif eri_pair_matrix is not None:
        pair = jnp.asarray(eri_pair_matrix)
        npair = nao * (nao + 1) // 2
        if pair.shape != (npair, npair):
            raise ValueError("eri_pair_matrix must be a square s4 AO pair matrix")
        rows, cols, _, _ = _metadata_arrays(nao, c.dtype)
        products = _mo_pair_products(c, c, rows, cols)
        other = _mo_pair_products(d, d, rows, cols)
        g = jnp.einsum(
            "pqP,PQ,rsQ->pqrs", products, pair, other, precision="highest"
        )
    else:
        factors = jnp.asarray(df_factors)
        if factors.ndim != 3 or factors.shape[1:] != (nao, nao):
            raise ValueError("df_factors must have shape (naux, nao, nao)")
        b = jnp.einsum("Lpq,pP,qQ->LPQ", factors, c, c, precision="highest")
        other = jnp.einsum("Lpq,pP,qQ->LPQ", factors, d, d, precision="highest")
        g = jnp.einsum("Lpq,Lrs->pqrs", b, other, precision="highest")
    return g


def validate_unrestricted_integrals(h1, eri):
    """h1=(ha,hb), eri=(gaa,gab,gbb), with gab[pqrs]=(pa qa|rb sb)."""
    if len(h1) != 2 or len(eri) != 3:
        raise ValueError("Unrestricted integrals require two h1 and three ERI blocks")
    ha, gaa = validate_integrals(h1[0], eri[0])
    hb, gbb = validate_integrals(h1[1], eri[2], ha.shape[0])
    _, gab = validate_integrals(ha, eri[1], ha.shape[0])
    return (ha, hb), (gaa, gab, gbb)


def transform_unrestricted_integrals(hcore, mo_coeff, **kwargs):
    """Real alpha/beta MO transform from full, s4 or density-fitted AO ERIs."""
    if len(mo_coeff) != 2:
        raise ValueError("Supply alpha and beta MO coefficients")
    ca, cb = map(jnp.asarray, mo_coeff)
    if ca.shape != cb.shape:
        raise ValueError("Alpha and beta MO spaces must have equal dimensions")
    ha, gaa = transform_integrals(hcore, ca, **kwargs)
    hb, gbb = transform_integrals(hcore, cb, **kwargs)
    gab = _transform_eri(ca, cb, **kwargs)
    return validate_unrestricted_integrals((ha, hb), (gaa, gab, gbb))


def unrestricted_frozen_indices(nmo, nocc, frozen=None):
    """A core count/shared list, or (alpha_list, beta_list) for separate spaces."""
    if len(nocc) != 2 or any(not isinstance(n, Integral) or isinstance(n, bool)
                             or not 0 <= n <= nmo for n in nocc):
        raise ValueError("nocc must be (nalpha, nbeta) with integer occupations")
    separate = (isinstance(frozen, (tuple, list)) and len(frozen) == 2
                and all(isinstance(f, (tuple, list)) for f in frozen))
    choices = frozen if separate else (frozen, frozen)
    return tuple(frozen_indices(nmo, n, f) for n, f in zip(nocc, choices))


def spin_orbital_integrals(h1, eri):
    """Dense real chemists' spin integrals; all alpha orbitals precede beta.

    Reference implementation: O((2*nmo)**4) storage. No spin-flip matrix
    elements are inserted. The beta-alpha block is the pair transpose of gab.
    """
    (ha, hb), (gaa, gab, gbb) = validate_unrestricted_integrals(h1, eri)
    n = ha.shape[0]
    dtype = jnp.result_type(ha, hb, gaa, gab, gbb)
    h = jnp.zeros((2*n, 2*n), dtype=dtype).at[:n, :n].set(ha).at[n:, n:].set(hb)
    g = jnp.zeros((2*n,)*4, dtype=dtype)
    g = g.at[:n, :n, :n, :n].set(gaa).at[:n, :n, n:, n:].set(gab)
    g = g.at[n:, n:, :n, :n].set(gab.transpose(2, 3, 0, 1))
    return h, g.at[n:, n:, n:, n:].set(gbb)


def frozen_indices(nmo, nocc, frozen):
    if frozen is None:
        return ()
    if isinstance(frozen, Integral) and not isinstance(frozen, bool):
        if frozen < 0 or frozen > nocc:
            raise ValueError("frozen core count must lie between 0 and nocc")
        return tuple(range(frozen))
    try:
        indices = tuple(frozen)
    except TypeError as error:
        raise ValueError("frozen must be a core count or orbital index list") from error
    if any(
        not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= nmo
        for i in indices
    ):
        raise ValueError("Frozen orbital indices must be integers in [0, nmo)")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate frozen orbital indices")
    return tuple(sorted(int(i) for i in indices))
