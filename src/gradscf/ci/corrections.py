"""Head-Gordon singlet CIS(D); distinct from variational CISD."""
from types import SimpleNamespace

import jax.numpy as jnp

from ..tddft.cisd import restricted_cisd_second_order_correction
from ..tddft.types import TDAResult
from .space import frozen_indices
from .types import CISDCorrectionResult
from ..solvers.diagnostics import require_converged_derivative


def cis_d_correction(eri, mo_energy, singles, *, nocc, frozen=None, denominator_tol=1e-10):
    """Return unscaled singlet CIS(D) corrections, in Hartree.

    Inputs must describe canonical RHF orbitals and singlet CIS roots. Spatial
    amplitudes use solve_cis's unit norm. No empirical scaling, damping or level
    shift is applied. Near-singular MP2/excited denominators are marked invalid.

    For derivatives, solve_cis must use gradient_mode='implicit_eigenvector';
    otherwise this function reports NaN correction derivatives, not partial AD.
    Formula: <CIS|V|U2 HF> + <CIS|V|T2 U1 HF> - E_MP2.
    Method: Head-Gordon et al. (1994), doi:10.1016/0009-2614(94)00070-0.
    Working-equation companion (not a numerical software comparison):
    https://manual.q-chem.com/5.1/sect-excorr.html (7.38–7.40).
    See ci/REFERENCES.md for attribution and validation scope.
    """
    if isinstance(singles.singlet, bool) and not singles.singlet:
        raise NotImplementedError("CIS(D) currently requires singlet CIS roots")
    eri, eps = jnp.asarray(eri), jnp.asarray(mo_energy)
    if jnp.iscomplexobj(eri) or jnp.iscomplexobj(eps):
        raise NotImplementedError("CIS(D) requires real integrals and orbital energies")
    if eps.ndim != 1 or eri.shape != (eps.size,) * 4:
        raise ValueError("Inconsistent MO energies and ERI dimensions")
    if not isinstance(nocc, int) or not 0 < nocc < eps.size:
        raise ValueError("CIS(D) requires occupied and virtual orbitals")
    if denominator_tol <= 0:
        raise ValueError("denominator_tol must be positive")
    frozen = frozen_indices(eps.size, nocc, frozen)
    occ = [i for i in range(nocc) if i not in frozen]
    vir = [i for i in range(nocc, eps.size) if i not in frozen]
    if not occ or not vir:
        raise ValueError("CIS(D) requires active occupied and virtual orbitals")
    if singles.amplitudes.shape != (singles.excitation_energies.size, len(occ), len(vir)):
        raise ValueError("CIS amplitudes do not match the active orbital space")
    active = jnp.asarray(occ + vir, dtype=jnp.int32)
    eps = eps[active]
    g = eri[jnp.ix_(active, active, active, active)]
    no, nm = len(occ), len(occ) + len(vir)
    denominator = (eps[None, None, no:, None] + eps[None, None, None, no:]
                   - eps[:no, None, None, None] - eps[None, :no, None, None])
    omegas = singles.excitation_energies
    min_abs = jnp.minimum(jnp.min(jnp.abs(denominator)), jnp.min(
        jnp.abs(denominator[None, ...] - omegas[:, None, None, None, None]), axis=(1, 2, 3, 4)))
    reference = SimpleNamespace(mo_coeff=jnp.eye(nm, dtype=g.dtype), mo_energy=eps,
                                mo_occ=jnp.concatenate([jnp.full((no,), 2.), jnp.zeros((nm-no,))]),
                                mo_eri=g, nocc=no)
    # The legacy restricted-TDA convention has ||X||^2=1/2.
    tda = TDAResult(omegas, singles.amplitudes / jnp.sqrt(2.))
    raw = restricted_cisd_second_order_correction(reference, tda)
    norm = jnp.sum(singles.amplitudes**2, axis=(1, 2))
    valid = (singles.converged & singles.singlet & (min_abs > denominator_tol) & jnp.isfinite(raw)
             & (jnp.abs(norm - 1) < 1e-7))
    raw = require_converged_derivative(raw, valid & singles.amplitude_response)
    correction = jnp.where(valid, raw, jnp.nan)
    return CISDCorrectionResult(omegas + correction, correction, omegas, min_abs, valid)
