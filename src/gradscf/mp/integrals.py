"""Canonical MP blocks; frozen cores still contribute to the HF Fock and energy."""
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from ..integrals.mo import (validate_integrals, validate_unrestricted_integrals,
                            frozen_indices, unrestricted_frozen_indices, transform_eri_block)


class MPIntegrals(NamedTuple):
    energies: tuple
    nocc: tuple
    blocks: object
    factors: object
    reference_energy: object
    canonical_error: object
    finite: object


def active_indices(nmo, nocc, frozen):
    frozen = frozen_indices(nmo, nocc, frozen)
    return (jnp.asarray([i for i in range(nocc) if i not in frozen], dtype=jnp.int32),
            jnp.asarray([a for a in range(nocc, nmo) if a not in frozen], dtype=jnp.int32))


def canonical_error(focks):
    return jnp.max(jnp.stack([jnp.max(jnp.abs(f - jnp.diag(jnp.diag(f))), initial=0.)
                              for f in focks]))


def prepare_integrals(h1, eri, *, nocc, frozen=None, nuclear_repulsion=0.):
    if isinstance(nocc, tuple):
        (ha, hb), (gaa, gab, gbb) = validate_unrestricted_integrals(h1, eri)
        n = ha.shape[0]
        fr = unrestricted_frozen_indices(n, nocc, frozen)
        na, nb = nocc
        fa = (ha + jnp.einsum("pqii->pq", gaa[:, :, :na, :na])
              - jnp.einsum("piiq->pq", gaa[:, :na, :na, :])
              + jnp.einsum("pqii->pq", gab[:, :, :nb, :nb]))
        fb = (hb + jnp.einsum("pqii->pq", gbb[:, :, :nb, :nb])
              - jnp.einsum("piiq->pq", gbb[:, :nb, :nb, :])
              + jnp.einsum("iipq->pq", gab[:na, :na]))
        energy = .5 * (jnp.trace((ha + fa)[:na, :na])
                       + jnp.trace((hb + fb)[:nb, :nb])) + nuclear_repulsion
        selections = tuple(active_indices(n, no, f) for no, f in zip(nocc, fr))
        (oa, va), (ob, vb) = selections
        blocks = (gaa[jnp.ix_(oa, va, oa, va)], gab[jnp.ix_(oa, va, ob, vb)],
                  gbb[jnp.ix_(ob, vb, ob, vb)])
        focks, gs = (fa, fb), (gaa, gab, gbb)
    else:
        h, g = validate_integrals(h1, eri)
        if isinstance(nocc, bool) or not isinstance(nocc, int) or not 0 <= nocc <= h.shape[0]:
            raise ValueError("nocc must be an integer between 0 and nmo")
        f = (h + 2 * jnp.einsum("pqii->pq", g[:, :, :nocc, :nocc])
             - jnp.einsum("piiq->pq", g[:, :nocc, :nocc, :]))
        energy = jnp.trace((h + f)[:nocc, :nocc]) + nuclear_repulsion
        selections = (active_indices(h.shape[0], nocc, frozen),)
        o, v = selections[0]
        blocks = (g[jnp.ix_(o, v, o, v)],)
        focks, gs = (f,), (g,)
    energies = tuple(jnp.diag(f)[jnp.concatenate((o, v))]
                     for f, (o, v) in zip(focks, selections))
    finite = (jnp.isfinite(energy) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
               for x in (*focks, *gs)])))
    return MPIntegrals(energies, tuple(len(o) for o, _ in selections), blocks,
                       None, energy, canonical_error(focks), finite)


def from_scf(source, *, frozen=None):
    """Reuse converged HF AO data; DF retains only Q/occupied/virtual factors.

    This is an eager adapter. It does not rerun SCF or transform complete MO ERIs.
    """
    from ..scf.facade import RKS, UKS
    from ..scf.rks import _build_jk
    from ..df import build_jk_from_df
    if not isinstance(source, (RKS, UKS)) or str(source.xc).lower() != "hf":
        raise NotImplementedError("Canonical MP requires RHF/UHF; ROHF and DFT are not supported")
    source._check_reference_source()
    coeff = jnp.asarray(source.mo_coeff)
    if jnp.iscomplexobj(coeff):
        raise NotImplementedError("MP requires real orbitals")
    unrestricted = isinstance(source, UKS)
    if unrestricted:
        ref = source.to_reference()
        h, g, factors = ref.h1e, ref.rep_tensor, ref.df_factors
        ns = (ref.nocc_alpha, ref.nocc_beta)
        cs = tuple(coeff)
        density = jnp.asarray(ref.rdm1)
        jk = (lambda d: build_jk_from_df(factors, d)) if factors is not None else (lambda d: _build_jk(g, d))
        ja, ka = jk(density[0])
        jb, kb = jk(density[1])
        focks = (cs[0].T @ (h + ja + jb - ka) @ cs[0],
                 cs[1].T @ (h + ja + jb - kb) @ cs[1])
        fr = unrestricted_frozen_indices(coeff.shape[-1], ns, frozen)
        kwargs = {"df_factors": factors} if factors is not None else {
            "eri" if jnp.ndim(g) == 4 else "eri_pair_matrix": g}
    else:
        inputs = source._scf_inputs
        ns = (inputs.nelectron // 2,)
        if inputs.nelectron % 2:
            raise ValueError("RMP requires an even electron count")
        cs = (coeff,)
        focks = (coeff.T @ source.scf_result.fock_matrix @ coeff,)
        factors = inputs.df_factors
        fr = (frozen_indices(coeff.shape[1], ns[0], frozen),)
        if factors is not None:
            kwargs = {"df_factors": factors}
        elif inputs.eri is not None:
            kwargs = {"eri": inputs.eri}
        else:
            pair = inputs.response_eri_pair_matrix()
            if pair is None:
                raise ValueError("MP requires two-electron integral data; direct-only SCF has no saved ERIs")
            kwargs = {"eri_pair_matrix": pair}
    expected = tuple((np.arange(c.shape[1]) < no).astype(int) for c, no in zip(cs, ns))
    expected = np.stack(expected) if unrestricted else 2 * expected[0]
    if not np.array_equal(np.asarray(source.mo_occ), expected):
        raise ValueError("MP requires integer occupations and occupied orbitals first")
    selections = tuple(active_indices(c.shape[1], no, f) for c, no, f in zip(cs, ns, fr))
    ovs = tuple((c[:, o], c[:, v]) for c, (o, v) in zip(cs, selections))
    energies = tuple(jnp.diag(f)[jnp.concatenate((o, v))]
                     for f, (o, v) in zip(focks, selections))
    pairs = ((0, 0), (0, 1), (1, 1)) if unrestricted else ((0, 0),)
    if factors is not None:
        ov_factors = tuple(jnp.einsum("Qpq,pi,qa->Qia", factors, o, v, precision="highest")
                           for o, v in ovs)
        blocks = None
        finite_data = (factors, *ov_factors)
    else:
        blocks = tuple(transform_eri_block((*ovs[a], *ovs[b]), **kwargs) for a, b in pairs)
        ov_factors = None
        finite_data = (*kwargs.values(), *blocks)
    finite = jnp.isfinite(source.e_tot) & jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(x)) for x in (*focks, *finite_data)]))
    return MPIntegrals(energies, tuple(len(o) for o, _ in selections), blocks,
                       ov_factors, jnp.asarray(source.e_tot), canonical_error(focks), finite)
