"""Finite-temperature, real restricted matrix scGW on Matsubara axes.

The iterated state is the full static Fock matrix and frequency-dependent
correlation self-energy, in a fixed orthonormal orbital basis. Each Dyson
G determines the density, Pi=2GG, Wc=(1-Pi)^-1 Pi and Sigma_c=-G Wc. The
chemical potential is solved from that same interacting G, not from pole
occupations. Reference subtraction and frequency grids are implemented in
:mod:`gradscf.gw.matsubara`.

This is a finite-grid imaginary-axis implementation. It returns matrix G
and Sigma, densities, internal energies, and residuals; quasiparticle poles
require a separate analytic-continuation calculation and are not returned.
The default solver is eager. An opt-in implicit response path compiles the
primal loop and differentiates the joint matrix/particle-number residual,
not the iterations. Inner kernels and response require resolvable reference
spectra; see SCGW.md for charge-conditioning and derivative limits.

References: Yeh et al., Phys. Rev. B 106, 235104 (2022);
Caruso et al., Phys. Rev. B 88, 075105 (2013); Galitskii and Migdal,
Sov. Phys. JETP 7, 96 (1958).
"""

from dataclasses import dataclass
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..df import build_jk_from_df
from ..scf._pytree import pytree_dataclass
from ..scf.autodiff import SCFDifferentiationConfig
from .g0w0 import _mo_factors
from .matsubara import MatsubaraGrid, matsubara_grid, dyson_green_and_density, gw_matsubara_step


@pytree_dataclass(static_fields=("nw", "beta"))
@dataclass(frozen=True)
class SCGWResult:
    """Converged imaginary-axis state; energies in Ha, beta in Ha^-1.

    ``mo_energy`` is None: no QP poles are inferred from Matsubara data.
    ``mo_coeff`` is the fixed AO expansion of the orthonormal basis used by
    ``green_iw``, ``self_energy_iw``, ``fock_mo``, and ``density_mo``. Full
    off-diagonal G carries the orbital response in that fixed basis.
    ``density_matrix`` is the spin-summed AO density; ``density_mo`` is its
    representation in the orthonormal basis. ``static_mo_*`` diagonalize
    the static Fock only and must not be interpreted as QP poles/orbitals.

    ``correlation_energy`` is the dynamical Sigma_c G contribution to the
    Galitskii-Migdal internal energy, not total_energy minus an HF energy.
    ``fixed_point_residual`` bounds unmixed Fock/Sigma/moment updates
    (the moment is expressed as Sigma at the lowest frequency), in Ha.
    The returned Fock/Sigma are the ones used in the returned Dyson G.
    The ``mapped_*`` fields are rebuilt from that same G; their difference
    from the Dyson inputs is bounded by the convergence residual. The GM
    energy is the GW functional of this returned G, with analytic reference
    tails in its bosonic Pi Wc sum.
    """

    mo_energy: jnp.ndarray | None
    mo_coeff: jnp.ndarray
    chemical_potential: jnp.ndarray
    correlation_energy: jnp.ndarray
    total_energy: jnp.ndarray
    density_matrix: jnp.ndarray
    density_mo: jnp.ndarray
    green_iw: jnp.ndarray
    self_energy_iw: jnp.ndarray
    sigma_moment: jnp.ndarray
    fock_mo: jnp.ndarray
    mapped_fock_mo: jnp.ndarray
    mapped_self_energy_iw: jnp.ndarray
    static_mo_energy: jnp.ndarray
    static_mo_coeff: jnp.ndarray
    fixed_point_residual: jnp.ndarray
    particle_number_error: jnp.ndarray
    grid: MatsubaraGrid
    converged: bool | jnp.ndarray
    nw: int
    n_iter: int | jnp.ndarray
    beta: float


def _solve_chemical_potential(fock, sigma, guess, grid, target, particle_tol):
    """Bracketed number solve with density from the interacting Dyson G."""
    def evaluate(mu):
        green, density = dyson_green_and_density(fock, sigma, mu, grid)
        count = float(2 * jnp.trace(density))
        if not np.isfinite(count):
            raise ArithmeticError("Nonfinite scGW density during the chemical-potential solve.")
        return count, green, density

    count, green, density = evaluate(guess)
    if abs(count - target) < particle_tol:
        return guess, green, density
    span = max(1.0, float(np.ptp(np.linalg.eigvalsh(np.asarray(fock)))), float(jnp.max(jnp.abs(sigma))))
    for _ in range(20):
        lower, upper = guess - span, guess + span
        nlower, _, _ = evaluate(lower)
        nupper, _, _ = evaluate(upper)
        if nlower <= target <= nupper:
            break
        span *= 2
    else:
        raise ArithmeticError("Could not bracket the scGW electron number; check the frequency grid and inputs.")
    for _ in range(80):
        mu = 0.5 * (lower + upper)
        count, green, density = evaluate(mu)
        if abs(count - target) < particle_tol:
            return mu, green, density
        if not nlower - particle_tol <= count <= nupper + particle_tol:
            raise ArithmeticError("Non-monotone scGW particle number on this grid; increase nw.")
        if count < target:
            lower, nlower = mu, count
        else:
            upper, nupper = mu, count
    raise ArithmeticError("scGW chemical potential did not meet particle_tol; increase nw or check the inputs.")


def _assemble_scgw_result(*, coeff, hcore, b, nuclear_repulsion, fock, sigma, moment,
                          mu, grid, nocc, n_iter, converged):
    """Recompute all observables from one state, retaining explicit input AD."""
    green, per_spin_density = dyson_green_and_density(fock, sigma, mu, grid)
    density = 2 * per_spin_density
    step = gw_matsubara_step(green, fock, mu, b, grid, moment)
    jmat, kmat = build_jk_from_df(b, density)
    new_fock = hcore + jmat - 0.5 * kmat
    residual = jnp.maximum(jnp.max(jnp.abs(new_fock - fock)), jnp.max(jnp.abs(step["sigma_iw"] - sigma)))
    residual = jnp.maximum(residual, jnp.max(jnp.abs(step["sigma_moment"] - moment)) / (jnp.pi / grid.beta))
    ec = step["correlation_energy"]
    e_one = jnp.einsum("ij,ji->", density, hcore)
    e_h = 0.5 * jnp.einsum("ij,ji->", density, jmat)
    e_x = -0.25 * jnp.einsum("ij,ji->", density, kmat)
    static_energy, static_rotation = jnp.linalg.eigh(fock)
    return SCGWResult(
        mo_energy=None, mo_coeff=coeff, chemical_potential=jnp.asarray(mu),
        correlation_energy=ec, total_energy=e_one + e_h + e_x + ec + nuclear_repulsion,
        density_matrix=coeff @ density @ coeff.T, density_mo=density,
        green_iw=green, self_energy_iw=sigma, sigma_moment=moment, fock_mo=fock,
        mapped_fock_mo=new_fock, mapped_self_energy_iw=step["sigma_iw"],
        static_mo_energy=static_energy, static_mo_coeff=coeff @ static_rotation,
        fixed_point_residual=jax.lax.stop_gradient(residual), particle_number_error=jnp.trace(density) - 2 * nocc,
        grid=grid, converged=converged, nw=grid.nw, n_iter=n_iter, beta=grid.beta,
    )


def scgw_matsubara_restricted(
    *,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
    df_factors: Array,
    hcore_matrix: Array,
    nuclear_repulsion: float = 0.0,
    nw: int = 100,
    beta: float = 100.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    mixing: float = 0.5,
    particle_tol: float = 1e-9,
    differentiation: SCFDifferentiationConfig | None = None,
    charge_response_tol: float = 1e-10,
) -> SCGWResult:
    """Solve the restricted matrix scGW equations at finite inverse temperature.

    Inputs use the molecular GW conventions. mo_coeff must span an
    orthonormal basis (C.T S C=I); hcore/DF factors are supplied in AO form.
    The input spectrum initializes mu; the occupied orbitals initialize
    density. Both G and the self-energy subsequently have full matrices.

    beta is in Ha^-1. nw counts positive fermionic frequencies (2*nw total),
    independent of beta. Increase nw at fixed beta before assessing the
    low-temperature limit by increasing beta. eta is not used on this axis.
    mixing applies to full Fock, Sigma(iw), and its high-frequency moment.
    Convergence uses unmixed residuals (Ha) and particle_tol (electrons).
    Results are finite-temperature internal energies, not free energies.

    differentiation optionally supplies SCFDifferentiationConfig(mode="implicit")
    to enable JIT, JVP and VJP with respect to real symmetric physical inputs.
    beta/nw/nocc and solver controls remain static. A failed response solve
    returns NaNs. charge_response_tol is the minimum resolved self-consistent
    charge susceptibility (electrons/Ha); mu is differentiated, never frozen.
    """
    physical = (mo_energy, mo_coeff, df_factors, hcore_matrix, nuclear_repulsion)
    if isinstance(beta, jax.core.Tracer) or (differentiation is None and any(
        isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(physical)
    )):
        raise NotImplementedError(
            "scGW default evaluation is eager; supply differentiation for implicit AD. beta must remain static."
        )
    if any(jnp.iscomplexobj(x) for x in physical):
        raise NotImplementedError("scGW currently supports real restricted molecular inputs only.")
    if max_iter < 1 or not np.isfinite(tol) or tol <= 0 or not np.isfinite(particle_tol) or particle_tol <= 0:
        raise ValueError("max_iter, tol and particle_tol must be positive and finite.")
    if not 0.0 < mixing <= 1.0:
        raise ValueError("mixing must be in (0, 1].")
    grid = matsubara_grid(nw=nw, beta=beta)
    coeff = jnp.asarray(mo_coeff, dtype=jnp.float64)
    energy = jnp.asarray(mo_energy, dtype=jnp.float64)
    nocc = int(nocc)
    if energy.ndim != 1 or coeff.ndim != 2 or coeff.shape[1] != energy.size:
        raise ValueError("mo_energy and mo_coeff shapes must describe the same orbital basis.")
    if not 0 < nocc < energy.size:
        raise ValueError("scGW requires both occupied and virtual orbitals: 0 < nocc < nmo.")
    hcore = coeff.T @ jnp.asarray(hcore_matrix, dtype=jnp.float64) @ coeff
    b = _mo_factors(jnp.asarray(df_factors, dtype=jnp.float64), coeff)
    # Physical real Hamiltonians and pair vertices are symmetric. This also
    # defines the symmetric-matrix tangent convention for response inputs.
    hcore = 0.5 * (hcore + hcore.T)
    b = 0.5 * (b + b.swapaxes(-1, -2))
    if differentiation is not None:
        from .scgw_response import implicit_scgw
        return implicit_scgw(
            coeff=coeff, energy_guess=energy, hcore=hcore, b=b, nocc=nocc,
            nuclear_repulsion=nuclear_repulsion, grid=grid, max_iter=int(max_iter),
            tol=float(tol), mixing=float(mixing), particle_tol=float(particle_tol),
            config=differentiation, charge_response_tol=charge_response_tol,
        )
    density = jnp.diag(jnp.where(jnp.arange(energy.size) < nocc, 2.0, 0.0))
    jmat, kmat = build_jk_from_df(b, density)
    fock = hcore + jmat - 0.5 * kmat
    sigma = jnp.zeros((2 * grid.nw, energy.size, energy.size), dtype=jnp.complex128)
    moment = jnp.zeros_like(fock)
    mu = float(0.5 * (energy[nocc - 1] + energy[nocc]))
    last_residual = np.inf

    for iteration in range(1, int(max_iter) + 1):
        mu, green, per_spin_density = _solve_chemical_potential(
            fock, sigma, mu, grid, 2 * nocc, particle_tol
        )
        density = 2 * per_spin_density
        step = gw_matsubara_step(green, fock, mu, b, grid, moment)
        new_sigma = step["sigma_iw"]
        new_moment = step["sigma_moment"]
        jmat, kmat = build_jk_from_df(b, density)
        new_fock = hcore + jmat - 0.5 * kmat
        residual = jnp.maximum(jnp.max(jnp.abs(new_fock - fock)), jnp.max(jnp.abs(new_sigma - sigma)))
        # The tail moment is part of the state: scale to its contribution
        # at the lowest Matsubara frequency so the residual is in Ha.
        residual = jnp.maximum(residual, jnp.max(jnp.abs(new_moment - moment)) / (jnp.pi / grid.beta))
        last_residual = float(residual)
        if not np.isfinite(last_residual):
            raise ArithmeticError("Nonfinite scGW update; check dielectric stability and grid resolution.")
        if last_residual < tol:
            return _assemble_scgw_result(
                coeff=coeff, hcore=hcore, b=b, nuclear_repulsion=nuclear_repulsion,
                fock=fock, sigma=sigma, moment=moment, mu=mu, grid=grid,
                nocc=nocc, n_iter=iteration, converged=True,
            )
        fock = (1 - mixing) * fock + mixing * new_fock
        sigma = (1 - mixing) * sigma + mixing * new_sigma
        moment = (1 - mixing) * moment + mixing * new_moment
    raise ArithmeticError(
        f"scGW did not converge in {max_iter} iterations "
        f"(unmixed matrix/tail residual {last_residual:.3e} Ha > {tol}). "
        "Increase max_iter, adjust mixing, and check nw/beta convergence."
    )


def scgw_cd_restricted(*, eta: float | None = None, **kwargs) -> SCGWResult:
    """Deprecated spelling of scgw_matsubara_restricted.

    The previous implementation was a pole approximation, not matrix scGW.
    This compatibility entry uses finite-temperature Matsubara grids; nw
    now counts positive fermionic frequencies and beta is explicit. No QP
    poles are returned. A supplied real-axis eta is rejected, not ignored.
    """
    if eta is not None:
        raise ValueError("eta is a real-frequency CD parameter; remove it when using Matsubara scGW.")
    warnings.warn(
        "scgw_cd_restricted now delegates to finite-temperature Matsubara scGW; "
        "use scgw_matsubara_restricted with explicit beta. mo_energy is None without analytic continuation.",
        DeprecationWarning, stacklevel=2,
    )
    return scgw_matsubara_restricted(**kwargs)


__all__ = ["scgw_matsubara_restricted", "scgw_cd_restricted", "SCGWResult"]
