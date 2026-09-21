"""Explicit real-orbital UHF/UKS stability analysis and bounded saddle escape.

These host-side diagnostics select a solution branch; they are not an AD rule.
Internal stability does not establish a global minimum or UHF->GHF stability.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .orbital_optimization import _problem_functions
from .uhf import UHFResult, run_uhf_from_integrals
from .uks import UKSConfig, UKSResult, run_uks_from_integrals


@dataclass(frozen=True)
class UnrestrictedStabilityResult:
    stable: bool | None
    minimum_curvature: float
    eigenvalues: jax.Array
    residual_norms: jax.Array
    eigensolver_converged: bool
    mo_coeff: jax.Array


@dataclass(frozen=True)
class UnrestrictedStabilizationResult:
    result: UHFResult | UKSResult
    stability: UnrestrictedStabilityResult
    restarts: int
    energy_history: tuple[float, ...]

    @property
    def stable(self):
        return self.stability.stable

    @property
    def minimum_curvature(self):
        return self.stability.minimum_curvature


def uks_stability(result: UKSResult | UHFResult, *, eri, ao, ao_deriv1,
                  grid_weights, config: UKSConfig, stability_tol=1e-5,
                  eigensolver_tol=1e-7, max_cycle=100) -> UnrestrictedStabilityResult:
    """Check internal unrestricted HF/KS curvature and return a proposed unit Cayley rotation.

    Curvature is d²E/dκ² for normalized occupied-virtual angle vectors (Hartree).
    Negative curvature below ``-stability_tol`` is unstable. ``stable=None``
    means SCF or the eigenpair residual check failed. No orbitals are mutated.
    The existing Davidson solver receives JAX HVPs, never a dense Hessian.
    """
    from ..solvers.eigen.davidson import _davidson_lowest_symmetric

    if config.jk_backend != 'full' or config.potential_clip is not None:
        raise NotImplementedError('Stability requires full ERIs and an unclipped XC potential.')
    if not jax.config.x64_enabled:
        raise ValueError('Unrestricted stability requires JAX float64.')
    if (not np.isfinite(stability_tol) or stability_tol <= 0 or
            not np.isfinite(eigensolver_tol) or eigensolver_tol <= 0 or max_cycle < 1):
        raise ValueError('Stability tolerances and max_cycle must be positive.')
    coeff = jnp.stack([result.mo_coeff_alpha, result.mo_coeff_beta])
    if jnp.iscomplexobj(coeff):
        raise ValueError('Internal unrestricted stability currently requires real orbitals.')
    empty = jnp.zeros(0, dtype=coeff.dtype)
    if not result.converged:
        return UnrestrictedStabilityResult(None, float('nan'), empty, empty, False, coeff)
    occ = np.stack([result.mo_occ_alpha, result.mo_occ_beta])
    dimension, rotate, _, _, energy = _problem_functions(
        'uks', config.xc_spec, 'col', tuple(map(tuple, occ)), uks_config=config)
    if dimension == 0:
        return UnrestrictedStabilityResult(True, float('inf'), empty, empty, True, coeff)
    args = dict(hcore=result.hcore_matrix, eri=jnp.asarray(eri),
                nuclear_repulsion=jnp.asarray(result.nuclear_repulsion),
                ao=jnp.asarray(ao), ao_deriv1=jnp.asarray(ao_deriv1),
                grid_weights=jnp.asarray(grid_weights))
    zero = jnp.zeros(dimension, dtype=coeff.dtype)
    # Linearizing the gradient retains the response intermediates across HVPs.
    _, hvp = jax.linearize(jax.grad(lambda angles: energy(angles, coeff, args)), zero)
    apply = jax.jit(jax.vmap(hvp, in_axes=1, out_axes=1))
    fock = jnp.stack([result.fock_matrix_alpha, result.fock_matrix_beta])
    eps = jnp.einsum('spi,spq,sqi->si', coeff, fock, coeff)
    gaps = []
    for spin in range(2):
        i, a = np.where(np.triu(occ[spin, :, None] != occ[spin, None, :], 1))
        gaps.append(2 * (eps[spin, a] - eps[spin, i]))
    values, vectors, converged = _davidson_lowest_symmetric(
        apply, size=dimension, diag=jnp.concatenate(gaps), nroots=min(3, dimension),
        tol=eigensolver_tol, max_iter=max_cycle, max_subspace=min(40, dimension),
        initial_guess_count=min(8, dimension), positive_eig_threshold=None)
    residuals = jnp.linalg.norm(apply(vectors) - vectors * values, axis=0)
    valid = bool(converged) and bool(jnp.all(jnp.isfinite(values))) and bool(
        jnp.all(residuals <= eigensolver_tol * 1.01))
    minimum = float(values[0])
    stable = minimum >= -stability_tol if valid else None
    proposed = rotate(vectors[:, 0], coeff) if stable is False else coeff
    return UnrestrictedStabilityResult(stable, minimum, values, residuals, valid, proposed)


# Keep the original UHF result names as compatibility aliases.
UHFStabilityResult = UnrestrictedStabilityResult
UHFStabilizationResult = UnrestrictedStabilizationResult


def uhf_stability(result: UHFResult, *, eri, **controls) -> UnrestrictedStabilityResult:
    """HF specialization of the shared unrestricted curvature check."""
    n = result.hcore_matrix.shape[0]
    return uks_stability(result, eri=eri, ao=jnp.zeros((0, n)),
        ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0),
        config=UKSConfig(xc_spec='hf'), **controls)


def stabilize_uhf_from_integrals(*, max_restarts=5, stability_tol=1e-5,
                                 eigensolver_tol=1e-7, **scf_inputs):
    """UHF specialization of the shared bounded stability/restart workflow."""
    return _stabilize(run_uhf_from_integrals,
        lambda result: uhf_stability(result, eri=scf_inputs['eri'],
            stability_tol=stability_tol, eigensolver_tol=eigensolver_tol),
        scf_inputs, max_restarts)


def stabilize_uks_from_integrals(*, max_restarts=5, stability_tol=1e-5,
                                 eigensolver_tol=1e-7, **scf_inputs):
    """SCF plus internal stability and directed restarts for LDA/GGA/hybrids.

    Takes the arguments of run_uks_from_integrals. Supply the identical grid
    and XC config for analysis and SCF. Only full-ERI, ordinary XC is supported.
    max_restarts=0 checks without restarting; unresolved checks return None.
    This discrete host-side branch selection is outside SCF differentiation.
    """
    if scf_inputs.get('bound_xc') is not None:
        raise NotImplementedError('Bound/Neural XC stability is not yet supported.')
    config = scf_inputs.get('config') or UKSConfig()
    if config.jk_backend != 'full' or config.potential_clip is not None:
        raise NotImplementedError('Stability requires full ERIs and an unclipped XC potential.')
    options = {key: scf_inputs[key] for key in ('eri', 'ao', 'ao_deriv1', 'grid_weights')}
    return _stabilize(run_uks_from_integrals,
        lambda result: uks_stability(result, **options, config=config,
            stability_tol=stability_tol, eigensolver_tol=eigensolver_tol),
        scf_inputs, max_restarts)


def _stabilize(solve, analyze, scf_inputs, max_restarts):
    if not isinstance(max_restarts, int) or max_restarts < 0:
        raise ValueError('max_restarts must be a nonnegative integer.')
    inputs = dict(scf_inputs)
    result = solve(**inputs)
    history = [float(result.total_energy)]
    for attempt in range(max_restarts + 1):
        check = analyze(result)
        if check.stable is not False or attempt == max_restarts:
            break
        occ = jnp.stack([result.mo_occ_alpha, result.mo_occ_beta])
        density = jnp.einsum('spi,si,sqi->spq', check.mo_coeff, occ, check.mo_coeff)
        inputs.update(init_density_alpha=density[0], init_density_beta=density[1])
        candidate = solve(**inputs)
        if not candidate.converged or not candidate.total_energy < result.total_energy - 1e-10:
            return UnrestrictedStabilizationResult(result, check, attempt + 1, tuple(history))
        result = candidate
        history.append(float(result.total_energy))
    return UnrestrictedStabilizationResult(result, check, attempt, tuple(history))
