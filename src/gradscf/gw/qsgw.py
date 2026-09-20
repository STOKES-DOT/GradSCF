"""Quasiparticle self-consistent GW (qsGW) -- Stage 4 (molecular, restricted).

qsGW replaces the dynamical self-energy by a static, Hermitian effective
potential and iterates orbitals + energies to self-consistency:

    v^qsGW_mn = (Re Sigma^c_mn(e_m) + Re Sigma^c_mn(e_n))/2  [mode A]
    F^qsGW     = h + J[rho] + Sigma^x + v^qsGW     [effective Fock]
    (C, e)     = eig(F^qsGW)  ->  rho -> iterate

Unlike G0W0/evGW there is no Dyson root search: the quasiparticle
energies are the eigenvalues of the effective Fock, and the whole
mean-field potential v^mf (v_xc or -K) is replaced by Sigma^x + v^qsGW.

The off-diagonal self-energy is evaluated with the contour-deformation
kernels (:func:`gradscf.gw.self_energy.sigma_imag_matrix` and
:func:`gradscf.gw.self_energy.sigma_residue_matrix`). The endpoint average
is the mode-A mapping of Kotani et al., PRB 76, 165106 (2007), Eq. (10).
This driver supports real molecular orbitals; Re is the Hermitian part
of the self-energy (elementwise real part for these symmetric matrices).

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
    """Mode-A endpoint average of the correlation self-energy (real MOs)."""
    nmo = mo_energy.shape[0]

    def response_fn(omega):
        return rho_response_iw(omega, mo_energy, b_ov, spin_factor=4.0)

    wmn_full = screened_w_imag_axis_matrix(b_mn, response_fn, freqs)  # (nw, nq, m, n)
    channels = ((mo_energy, b_ov, 2.0),)

    def at_endpoint(args):
        m, omega = args
        sigma = sigma_imag_matrix(omega, wmn_full, mo_energy, ef, freqs, wts, eta)
        sigma = sigma + sigma_residue_matrix(omega, mo_energy, b_mn, channels, ef, eta)
        # Re Sigma is the Hermitian part at this frequency. Retain only
        # the required row instead of stacking nmo full self-energy matrices.
        return 0.5 * (sigma[m] + sigma[:, m].conj()).real

    # One evaluation per orbital energy, rather than nmo**2 midpoints.
    rows = jax.lax.map(at_endpoint, (jnp.arange(nmo), mo_energy))
    return 0.5 * (rows + rows.T)


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
        Real mean-field starting orbitals/energies (HF or DFT).
    nocc:
        Number of doubly occupied orbitals.
    df_factors:
        Low-rank ERI factors ``(naux, nao, nao)``.
    hcore_matrix:
        One-electron core Hamiltonian (AO).
    nw, eta:
        Imaginary-grid size and broadening.
    max_iter, tol, tol_density, damping:
        Outer self-consistency controls. ``damping`` mixes eigenvalues;
        ``tol`` bounds the undamped energy update and the largest entry of
        ``F_MO - diag(energy)`` (Ha). ``tol_density`` bounds the density change.

    Returns
    -------
    :class:`gradscf.gw.types.GWResult` with updated ``mo_coeff``.

    Notes
    -----
    The Hartree term is rebuilt from the qsGW density each iteration; the
    starting point only enters through the initial orbitals.  AD through
    the outer loop is not wired yet (use finite differences on the eager
    driver); JAX transformations of the outer driver are rejected explicitly.
    The inner static-mapping kernel remains differentiable. Convergence on
    a chosen discrete CD grid is separate from basis/grid accuracy.
    """
    inputs = (mo_energy, mo_coeff, df_factors, hcore_matrix)
    if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(inputs)):
        raise NotImplementedError("qsGW currently supports eager outer iterations only; AD/JIT is not implemented.")
    if any(jnp.iscomplexobj(x) for x in inputs):
        raise NotImplementedError("qsGW currently supports real molecular inputs only.")
    if max_iter < 1 or tol <= 0.0 or tol_density <= 0.0:
        raise ValueError("max_iter, tol and tol_density must be positive.")
    mo_energy = jnp.asarray(mo_energy, dtype=jnp.float64)
    coeff = jnp.asarray(mo_coeff, dtype=jnp.float64)
    hcore = jnp.asarray(hcore_matrix, dtype=jnp.float64)
    df_factors = jnp.asarray(df_factors)
    nocc = int(nocc)
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
        d_fock = float(jnp.max(jnp.abs(fock_mo - jnp.diag(energy))))
        new_energy, rotation = jnp.linalg.eigh(fock_mo)
        new_coeff = coeff @ rotation
        # Damping can make the accepted step arbitrarily small even when
        # the actual fixed-point energy residual is still large.
        d_energy = float(jnp.max(jnp.abs(new_energy - energy)))
        if damping > 0.0:
            # Mix the Fock eigenvalues; orbitals follow the new rotation.
            new_energy = (1.0 - damping) * new_energy + damping * energy
        new_density = 2.0 * new_coeff[:, :nocc] @ new_coeff[:, :nocc].T
        d_density = float(jnp.max(jnp.abs(new_density - density)))
        last_delta = (d_energy, d_density, d_fock)
        if d_energy < tol and d_density < tol_density and d_fock < tol:
            # Return the state whose effective Fock was actually evaluated
            # and checked, not an unchecked next iterate.
            return GWResult(
                mo_energy=energy,
                mo_coeff=coeff,
                converged=True,
                sigma_qp=None,
                nw=int(nw),
            )
        energy, coeff = new_energy, new_coeff
    raise ArithmeticError(
        f"qsGW did not converge in {max_iter} iterations "
        f"(max|undamped de|={last_delta[0]:.3e}, max|drho|={last_delta[1]:.3e}, "
        f"max|F_MO-diag(e)|={last_delta[2]:.3e} Ha). "
        "Increase max_iter or damping; no fallback result is returned."
    )


__all__ = ["qsgw_cd_restricted"]
