"""Shared real-orbital RKS/UKS stability analysis and bounded saddle escape.

These host-side diagnostics select a solution branch; they are not an AD rule.
Internal stability does not establish a global minimum or UHF->GHF stability.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .orbital_optimization import _problem_functions, _rotation_functions
from ..solvers import LinearOperator, EigenSolverConfig, solve_hermitian
from .uhf import UHFResult, run_uhf_from_integrals
from .uks import UKSConfig, UKSResult, run_uks_from_integrals


@dataclass(frozen=True)
class OrbitalStabilityResult:
    stable: bool | None
    minimum_curvature: float
    eigenvalues: jax.Array
    residual_norms: jax.Array
    eigensolver_converged: bool
    mo_coeff: jax.Array
    gradient_norm: float = float('nan')
    stationary: bool = False
    spectrum_certified: bool = False


UnrestrictedStabilityResult = OrbitalStabilityResult


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


def _curvature_check(
    *,
    coeff,
    dimension,
    rotate,
    energy,
    args,
    diagonal,
    converged,
    stability_tol,
    eigensolver_tol,
    max_cycle,
    gradient_tol=1e-6,
    solver="davidson",
    max_space=40
):
    if not jax.config.x64_enabled:
        raise ValueError("Stability requires JAX float64")
    if (
        any(
            not np.isfinite(t) or t <= 0
            for t in (stability_tol, eigensolver_tol, gradient_tol)
        )
        or max_cycle < 1
        or max_space < 1
    ):
        raise ValueError("Stability tolerances and limits must be positive")
    if solver not in {"dense", "davidson"}:
        raise ValueError("Stability solver must be dense or davidson")
    empty = jnp.zeros(0, dtype=coeff.dtype)
    if not converged:
        return OrbitalStabilityResult(None, float("nan"), empty, empty, False, coeff)
    if dimension == 0:
        return OrbitalStabilityResult(
            True, float("inf"), empty, empty, True, coeff, 0.0, True, True
        )
    zero = jnp.zeros(dimension, dtype=coeff.dtype)
    gradient, hvp = jax.linearize(jax.grad(lambda x: energy(x, coeff, args)), zero)
    norm = float(0.5 * jnp.linalg.norm(gradient))
    if not np.isfinite(norm) or norm > gradient_tol:
        return OrbitalStabilityResult(
            None, float("nan"), empty, empty, False, coeff, norm
        )
    apply = jax.jit(jax.vmap(hvp, in_axes=1, out_axes=1))
    op = LinearOperator(
        (dimension, dimension), coeff.dtype, hvp, diagonal=diagonal, matmat=apply
    )
    solved = solve_hermitian(
        op,
        config=EigenSolverConfig(
            method=solver,
            nroots=min(3, dimension),
            atol=eigensolver_tol,
            maxiter=max_cycle,
            max_subspace=min(max_space, dimension),
            initial_guess_count=min(8, dimension),
        ),
    )
    values, vectors = solved.values, solved.vectors
    residuals = jnp.linalg.norm(apply(vectors) - vectors * values, axis=0)
    valid = bool(
        jnp.all(solved.converged)
        & jnp.all(jnp.isfinite(values))
        & jnp.all(residuals <= eigensolver_tol * 1.01)
    )
    minimum = float(values[0])
    stable = bool(minimum >= -stability_tol) if valid else None
    proposed = rotate(vectors[:, 0], coeff) if stable is False else coeff
    return OrbitalStabilityResult(
        stable,
        minimum,
        values,
        residuals,
        valid,
        proposed,
        norm,
        True,
        valid and solver == "dense",
    )


def uks_stability(
    result: UKSResult | UHFResult,
    *,
    eri,
    ao,
    ao_deriv1,
    grid_weights,
    config: UKSConfig,
    stability_tol=1e-5,
    eigensolver_tol=1e-7,
    max_cycle=100,
    gradient_tol=1e-6,
    solver="davidson",
    max_space=40
) -> UnrestrictedStabilityResult:
    """Real internal UHF/UKS curvature from the original spin energy evaluator.

    The shared public eigensolver sees HVP actions. Positive partial Ritz values
    are estimates, not global stability certificates. This host diagnostic does
    not require isolated-eigenvalue AD validity at a degenerate curvature.
    """
    if config.jk_backend != "full" or config.potential_clip is not None:
        raise NotImplementedError(
            "Stability requires full/packed ERIs and an unclipped XC potential."
        )
    coeff = jnp.stack([result.mo_coeff_alpha, result.mo_coeff_beta])
    if jnp.iscomplexobj(coeff):
        raise ValueError("Internal unrestricted stability requires real orbitals.")
    occ = np.stack([result.mo_occ_alpha, result.mo_occ_beta])
    dimension, rotate, _, _, energy = _problem_functions(
        "uks", config.xc_spec, "col", tuple(map(tuple, occ)), uks_config=config
    )
    args = dict(
        hcore=result.hcore_matrix,
        eri=jnp.asarray(eri),
        nuclear_repulsion=jnp.asarray(result.nuclear_repulsion),
        ao=jnp.asarray(ao),
        ao_deriv1=jnp.asarray(ao_deriv1),
        grid_weights=jnp.asarray(grid_weights),
    )
    fock = jnp.stack([result.fock_matrix_alpha, result.fock_matrix_beta])
    eps = jnp.einsum("spi,spq,sqi->si", coeff, fock, coeff)
    gaps = []
    for spin in range(2):
        i, a = np.where(np.triu(occ[spin, :, None] != occ[spin, None, :], 1))
        gaps.append(2 * (occ[spin, i] - occ[spin, a]) * (eps[spin, a] - eps[spin, i]))
    return _curvature_check(
        coeff=coeff,
        dimension=dimension,
        rotate=rotate,
        energy=energy,
        args=args,
        diagonal=jnp.concatenate(gaps),
        converged=result.converged,
        stability_tol=stability_tol,
        eigensolver_tol=eigensolver_tol,
        max_cycle=max_cycle,
        gradient_tol=gradient_tol,
        solver=solver,
        max_space=max_space,
    )


def restricted_stability(
    source,
    *,
    stability_tol=1e-5,
    eigensolver_tol=1e-7,
    max_cycle=100,
    gradient_tol=None,
    solver="davidson",
    max_space=40
):
    """Real restricted-internal curvature; does not test spin/complex instabilities."""
    from .diagnostics import _restricted_scf_state
    from .rks import _make_jk_builder, _energy_and_raw_fock_for_density
    from ..dft.libxc_jax.jax_libxc import hybrid_coeff, xc_type

    result, inputs = _restricted_scf_state(source)
    cfg = source._config()
    if cfg.jk_backend != "full" or cfg.potential_clip is not None:
        raise NotImplementedError(
            "Restricted stability requires full/packed ERIs and unclipped XC"
        )
    occ = np.asarray(result.mo_occ)
    if not np.all((occ == 0) | (occ == 2)):
        raise ValueError("Restricted stability requires integer 0/2 occupations")
    topology = tuple(map(tuple, np.stack([occ / 2, occ / 2])))
    dimension, rotate, density_from = _rotation_functions("roks", topology)
    pair = None if inputs.eri is not None else inputs.response_eri_pair_matrix()
    if inputs.eri is None and pair is None:
        raise ValueError("Restricted stability requires available AO ERIs")
    alpha = jnp.asarray(hybrid_coeff(cfg.xc_spec), dtype=result.mo_coeff.dtype)
    kind = xc_type(cfg.xc_spec)
    if kind not in {"HF", "LDA", "GGA"}:
        raise NotImplementedError(
            "Restricted stability supports HF/LDA/GGA/global hybrids"
        )
    jk = _make_jk_builder(
        inputs.eri,
        cfg,
        eri_pair_matrix=pair,
        with_k=abs(float(alpha)) > 1e-14,
        nocc=inputs.nelectron // 2,
    )

    def energy(angles, base, args):
        c = rotate(angles, base)
        density = jnp.sum(density_from(c), axis=0)
        return _energy_and_raw_fock_for_density(
            density=density,
            mo_coeff=c[0],
            mo_occ=result.mo_occ,
            mo_energy=result.mo_energy,
            jk_builder=jk,
            ao=inputs.ao,
            ao_deriv1=inputs.ao_deriv1,
            weights=inputs.grid_weights,
            h=inputs.hcore,
            enuc=inputs.nuclear_repulsion,
            alpha=alpha,
            cfg=cfg,
            xc_kind=kind,
        )[0]

    c = result.mo_coeff
    eps = jnp.diag(c.T @ result.fock_matrix @ c)
    i, a = np.where(np.triu(occ[:, None] != occ[None, :], 1))
    diagonal = 2 * (occ[i] - occ[a]) * (eps[a] - eps[i])
    check = _curvature_check(
        coeff=c[None],
        dimension=dimension,
        rotate=rotate,
        energy=energy,
        args={},
        diagonal=diagonal,
        converged=bool(source.converged and result.converged),
        stability_tol=stability_tol,
        eigensolver_tol=eigensolver_tol,
        max_cycle=max_cycle,
        gradient_tol=source.conv_tol_grad if gradient_tol is None else gradient_tol,
        solver=solver,
        max_space=max_space,
    )
    from dataclasses import replace

    return replace(check, mo_coeff=check.mo_coeff[0])


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
