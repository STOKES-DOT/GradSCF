"""Fully self-consistent GW (scGW) -- Stage 5 (molecular, restricted).

Matrix scGW on the imaginary axis:

    G(iw)  = [ (iw + mu) I - H0 - Sigma^c(iw) ]^{-1},   H0 = h + J[rho] + Sigma^x
    Sigma^c(iw) from the RPA screened interaction W(iw)
    mu adjusted so that the pole occupations integrate to N electrons

Pragmatic simplifications (documented; small-molecule validation scope):

1. The interacting density is approximated by the quasiparticle pole
   occupations, rho = C diag(2 f(e_m - mu)) C^T (no Matsubara tail sum);
   this keeps particle-number conservation exact without tail corrections.
2. The RPA response entering W uses the quasiparticle pole energies of the
   current iteration (diagonal-pole approximation of G).
3. Correlation energy via the Galitskii-Migdal trace on the imaginary
   grid, E_c = (1/2pi) sum_w wts_w Tr[Sigma^c(iw) G(iw)] (spin factor 2
   included), which converges on the scaled-Legendre grid without tails.

References
----------
- V. M. Galitskii and A. B. Migdal, Sov. Phys. JETP 7, 96 (1958)
  (Galitskii-Migdal total energy).
- B. Holm and U. von Barth, "Fully self-consistent GW self-energy of the
  electron gas", Phys. Rev. B 57, 2108 (1998). DOI:10.1103/PhysRevB.57.2108
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- A. Kutepov, "Electronic structure of Na, K, Si, and LiF from
  self-consistent GW", Phys. Rev. B 95, 195120 (2017).
  DOI:10.1103/PhysRevB.95.195120
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..df import build_j_from_df, build_jk_from_df
from .freq import scaled_legendre_grid
from .g0w0 import _exchange_mo, _mo_factors
from .polarizability import rho_response_iw
from .screened import screened_w_imag_axis_matrix
from .self_energy import sigma_imag_matrix
from ..scf._pytree import pytree_dataclass


@pytree_dataclass(static_fields=("converged", "nw", "n_iter"))
@dataclass(frozen=True)
class SCGWResult:
    """Result of a molecular scGW calculation."""

    mo_energy: jnp.ndarray  # quasiparticle pole energies
    mo_coeff: jnp.ndarray
    chemical_potential: jnp.ndarray
    correlation_energy: jnp.ndarray
    total_energy: jnp.ndarray
    converged: bool
    nw: int
    n_iter: int


def _fermi(energy: Array, mu: Array, beta: float = 1e5) -> Array:
    """T -> 0 Fermi occupation (smooth step for JAX friendliness)."""
    return 0.5 * (1.0 - jnp.tanh(0.5 * beta * (energy - mu)))


def scgw_cd_restricted(
    *,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
    df_factors: Array,
    hcore_matrix: Array,
    nuclear_repulsion: float = 0.0,
    nw: int = 100,
    eta: float = 1e-3,
    max_iter: int = 20,
    tol: float = 1e-6,
    mixing: float = 0.5,
) -> SCGWResult:
    """Spin-restricted matrix scGW with contour-deformation self-energy.

    Parameters follow :func:`gradscf.gw.qsgw_cd_restricted` plus
    ``mixing`` (fraction of the new pole energies accepted per iteration).
    Returns :class:`SCGWResult` including the Galitskii-Migdal total
    energy.  Raises ``ArithmeticError`` on non-convergence (no fallback).
    """
    mo_energy = jnp.asarray(mo_energy, dtype=jnp.float64)
    coeff = jnp.asarray(mo_coeff, dtype=jnp.float64)
    hcore = jnp.asarray(hcore_matrix, dtype=jnp.float64)
    df_factors = jnp.asarray(df_factors)
    nocc = int(nocc)
    nmo = mo_energy.shape[0]
    if not 0.0 < mixing <= 1.0:
        raise ValueError("mixing must be in (0, 1].")

    freqs, wts = scaled_legendre_grid(nw)
    eye = jnp.eye(nmo, dtype=jnp.complex128)
    energy = mo_energy
    mu = 0.5 * (energy[nocc - 1] + energy[nocc])
    h0_static = None
    last_delta = None

    for iteration in range(1, int(max_iter) + 1):
        b_mn = _mo_factors(df_factors, coeff)
        b_ov = b_mn[:, :nocc, nocc:]

        # Hartree + exchange from the pole-occupation density
        occ = 2.0 * _fermi(energy, mu)
        occ_coeff = coeff * jnp.sqrt(occ)[None, :]
        density = occ_coeff @ occ_coeff.T
        j_mat = build_j_from_df(df_factors, density)
        vx_mo = _exchange_mo(b_mn, nocc)
        h0_static = coeff.T @ (hcore + j_mat) @ coeff + vx_mo

        # Screened interaction from the current pole energies
        def response_fn(omega):
            return rho_response_iw(omega, energy, b_ov, spin_factor=4.0)

        wmn_full = screened_w_imag_axis_matrix(b_mn, response_fn, freqs)

        # Self-energy on the grid (vectorized over frequencies)
        ef = mu
        sigma_w = jax.vmap(
            lambda w: sigma_imag_matrix(w, wmn_full, energy, ef, freqs, wts, eta)
        )(freqs)

        # Dyson Green's function on the grid
        def dyson(w, sig):
            return jnp.linalg.solve((1j * w + mu) * eye - h0_static - sig, eye)

        g_w = jax.vmap(dyson)(freqs, sigma_w)

        # New pole energies: diagonal of H0 + Re Sigma at the old poles
        sigma_at_poles = jax.vmap(
            lambda m: sigma_imag_matrix(
                energy[m], wmn_full, energy, ef, freqs, wts, eta
            )[m, m]
        )(jnp.arange(nmo))
        new_energy = (h0_static + jnp.diag(sigma_at_poles)).diagonal().real
        # Chemical potential: midgap of the new spectrum (gap system)
        new_mu = 0.5 * (new_energy[nocc - 1] + new_energy[nocc])

        delta = float(jnp.max(jnp.abs(new_energy - energy)))
        last_delta = delta
        energy = (1.0 - mixing) * energy + mixing * new_energy
        mu = (1.0 - mixing) * mu + mixing * new_mu
        if delta < tol:
            # Galitskii-Migdal correlation energy on the final grid
            tr_sg = jnp.einsum("wmn,wnm->w", sigma_w, g_w, precision=Precision.HIGHEST)
            e_c = (1.0 / (2.0 * jnp.pi)) * jnp.sum(wts * tr_sg)
            e_c = 2.0 * e_c.real  # spin factor
            e_one = jnp.einsum("pq,pq->", density, hcore, precision=Precision.HIGHEST)
            e_h = 0.5 * jnp.einsum("pq,pq->", density, j_mat, precision=Precision.HIGHEST)
            _, k_tot = build_jk_from_df(df_factors, density)
            e_x = -0.25 * jnp.trace(k_tot @ density)
            total = e_one + e_h + e_x + e_c + float(nuclear_repulsion)
            return SCGWResult(
                mo_energy=energy,
                mo_coeff=coeff,
                chemical_potential=mu,
                correlation_energy=e_c,
                total_energy=jnp.asarray(total),
                converged=True,
                nw=int(nw),
                n_iter=iteration,
            )
    raise ArithmeticError(
        f"scGW did not converge in {max_iter} iterations "
        f"(last max|de| = {last_delta:.3e} > {tol}). "
        "Increase max_iter or reduce mixing; no fallback result is returned."
    )


__all__ = ["scgw_cd_restricted", "SCGWResult"]
