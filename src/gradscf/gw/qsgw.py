"""Quasiparticle self-consistent GW (qsGW) -- Stage 4 (molecular, restricted).

qsGW replaces the dynamical self-energy by a static, Hermitian effective
potential and iterates orbitals + energies to self-consistency:

    v^qsGW_mn = Re Sigma^c_mn( (e_m + e_n)/2 )     [static mapping]
    F^qsGW     = h + J[rho] + Sigma^x + v^qsGW     [effective Fock]
    (C, e)     = eig(F^qsGW)  ->  rho -> iterate

Unlike G0W0/evGW there is no Dyson root search: the quasiparticle
energies are the eigenvalues of the effective Fock, and the whole
mean-field potential v^mf (v_xc or -K) is replaced by Sigma^x + v^qsGW.

The off-diagonal self-energy is evaluated with the contour-deformation
kernels (:func:`gradscf.gw.self_energy.sigma_imag_matrix` and
:func:`gradscf.gw.self_energy.sigma_residue_matrix`); the midpoint
frequency follows the standard qsGW prescription.

References
----------
- S. V. Faleev, M. van Schilfgaarde and T. Kotani, "All-electron
  self-consistent GW approximation: Application to Si, MnO, and NiO",
  Phys. Rev. Lett. 93, 126406 (2004). DOI:10.1103/PhysRevLett.93.126406
- M. van Schilfgaarde, T. Kotani and S. V. Faleev, "Quasiparticle
  self-consistent GW theory", Phys. Rev. B 74, 245125 (2006).
  DOI:10.1103/PhysRevB.74.245125
- T. Kotani, M. van Schilfgaarde and S. V. Faleev, "Quasiparticle
  self-consistent GW method: a basis for the independent-particle
  approximation", Phys. Rev. B 76, 165106 (2007).
  DOI:10.1103/PhysRevB.76.165106
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..df import build_j_from_df
from .freq import scaled_legendre_grid
from .g0w0 import _exchange_mo, _mo_factors
from .polarizability import rho_response_iw
from .screened import screened_w_imag_axis_matrix
from .self_energy import sigma_imag_matrix, sigma_residue_matrix
from .types import GWResult


def _static_self_energy(
    *,
    b_mn: Array,
    b_ov: Array,
    mo_energy: Array,
    nocc: int,
    ef: Array,
    freqs: Array,
    wts: Array,
    eta: float,
) -> Array:
    """Static Hermitian correlation potential v^qsGW_mn = Re Sigma^c_mn(midpoint)."""
    nmo = mo_energy.shape[0]

    def response_fn(omega):
        return rho_response_iw(omega, mo_energy, b_ov, spin_factor=4.0)

    wmn_full = screened_w_imag_axis_matrix(b_mn, response_fn, freqs)  # (nw, nq, m, n)
    channels = ((mo_energy, b_ov, 2.0),)

    mid = 0.5 * (mo_energy[:, None] + mo_energy[None, :])  # (nmo, nmo)
    sigma = jnp.zeros((nmo, nmo), dtype=jnp.complex128)
    # Group the midpoint evaluations by row to reuse one residue-matrix
    # evaluation per row frequency batch; each call costs nq batched solves.
    for m in range(nmo):
        row_freqs = mid[m]  # (nmo,)
        imag_parts = jax.vmap(
            lambda w: sigma_imag_matrix(w, wmn_full, mo_energy, ef, freqs, wts, eta)
        )(row_freqs)
        res_parts = jax.vmap(
            lambda w: sigma_residue_matrix(w, mo_energy, b_mn, channels, ef, eta)
        )(row_freqs)
        sigma = sigma.at[m].set(
            imag_parts[jnp.arange(nmo), m, jnp.arange(nmo)]
            + res_parts[jnp.arange(nmo), m, jnp.arange(nmo)]
        )
    return 0.5 * (sigma + sigma.conj().T).real


def qsgw_cd_restricted(
    *,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
    df_factors: Array,
    hcore_matrix: Array,
    nw: int = 100,
    eta: float = 1e-3,
    max_iter: int = 20,
    tol: float = 1e-6,
    tol_density: float = 1e-5,
    damping: float = 0.0,
) -> GWResult:
    """Spin-restricted qsGW with contour deformation.

    Parameters
    ----------
    mo_energy, mo_coeff:
        Mean-field starting orbitals/energies (HF or DFT).
    nocc:
        Number of doubly occupied orbitals.
    df_factors:
        Low-rank ERI factors ``(naux, nao, nao)``.
    hcore_matrix:
        One-electron core Hamiltonian (AO).
    nw, eta:
        Imaginary-grid size and broadening.
    max_iter, tol, tol_density, damping:
        Outer self-consistency controls.

    Returns
    -------
    :class:`gradscf.gw.types.GWResult` with updated ``mo_coeff``.

    Notes
    -----
    The Hartree term is rebuilt from the qsGW density each iteration; the
    starting point only enters through the initial orbitals.  AD through
    the outer loop is not wired yet (use finite differences on the eager
    driver); the inner kernels remain differentiable.
    """
    mo_energy = jnp.asarray(mo_energy, dtype=jnp.float64)
    coeff = jnp.asarray(mo_coeff, dtype=jnp.float64)
    hcore = jnp.asarray(hcore_matrix, dtype=jnp.float64)
    df_factors = jnp.asarray(df_factors)
    nocc = int(nocc)
    nmo = mo_energy.shape[0]
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1).")

    freqs, wts = scaled_legendre_grid(nw)
    energy = mo_energy
    last_delta = None
    for _ in range(int(max_iter)):
        b_mn = _mo_factors(df_factors, coeff)
        b_ov = b_mn[:, :nocc, nocc:]
        occ_coeff = coeff[:, :nocc]
        density = 2.0 * occ_coeff @ occ_coeff.T
        j_mat = build_j_from_df(df_factors, density)
        vx_mo = _exchange_mo(b_mn, nocc)

        ef = 0.5 * (energy[nocc - 1] + energy[nocc])
        v_qsgw = _static_self_energy(
            b_mn=b_mn, b_ov=b_ov, mo_energy=energy, nocc=nocc, ef=ef,
            freqs=freqs, wts=wts, eta=float(eta),
        )
        fock_mo = coeff.T @ (hcore + j_mat) @ coeff + vx_mo + v_qsgw
        fock_mo = 0.5 * (fock_mo + fock_mo.T)
        new_energy, rotation = jnp.linalg.eigh(fock_mo)
        new_coeff = coeff @ rotation
        if damping > 0.0:
            # Mix the Fock eigenvalues; orbitals follow the new rotation.
            new_energy = (1.0 - damping) * new_energy + damping * energy
        new_density = 2.0 * new_coeff[:, :nocc] @ new_coeff[:, :nocc].T
        d_energy = float(jnp.max(jnp.abs(new_energy - energy)))
        d_density = float(jnp.max(jnp.abs(new_density - density)))
        last_delta = (d_energy, d_density)
        energy, coeff = new_energy, new_coeff
        if d_energy < tol and d_density < tol_density:
            return GWResult(
                mo_energy=energy,
                mo_coeff=coeff,
                converged=True,
                sigma_qp=None,
                nw=int(nw),
            )
    raise ArithmeticError(
        f"qsGW did not converge in {max_iter} iterations "
        f"(max|de|={last_delta[0]:.3e}, max|drho|={last_delta[1]:.3e}). "
        "Increase max_iter or damping; no fallback result is returned."
    )


__all__ = ["qsgw_cd_restricted"]
