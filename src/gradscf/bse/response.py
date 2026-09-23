"""Static physical problem assembly through GradSCF's common eigensolver."""

import jax.numpy as jnp
from ..solvers import (
    EigenSolverConfig,
    EigenResponseConfig,
    LinearSolverConfig,
    solve_hermitian,
    solve_rpa,
)
from ..solvers.diagnostics import require_converged_derivative
from ..gw.screened import build_static_screening
from .space import make_bse_space
from .types import BSEConfig, BSEResult
from .kernel import build_tda_operator, build_bse_operators


def run_bse(
    qp_energy,
    screening_energy,
    mo_factors,
    space,
    *,
    screening_space=None,
    qp_computed_mask=None,
    qp_converged_mask=None,
    config=None,
):
    """Real static BSE with explicit QP and screening spectra.

    Unspecified QP masks mean explicitly supplied energies, not inferred GW
    convergence. GW adapters must provide actual computed/converged masks.
    Requested isolated roots include an extra guard root for response checks.
    """
    cfg = BSEConfig() if config is None else config
    qp, e, l = map(jnp.asarray, (qp_energy, screening_energy, mo_factors))
    if (
        e.shape != (space.nmo,)
        or qp.shape != e.shape
        or l.ndim != 3
        or l.shape[1:] != (space.nmo,) * 2
    ):
        raise ValueError("BSE input shapes do not match the orbital space")
    if jnp.iscomplexobj(qp) or jnp.iscomplexobj(e) or jnp.iscomplexobj(l):
        raise NotImplementedError("Molecular BSE requires real closed-shell data")
    if l.size > cfg.max_factor_elements:
        raise ValueError("BSE factors exceed max_factor_elements")
    if not space.size or cfg.nroots > space.size:
        raise ValueError("Invalid BSE root count or empty excitation space")
    if cfg.solver == "dense" and space.size > cfg.max_dense:
        raise ValueError("BSE dense oracle exceeds max_dense")
    screen = (
        make_bse_space(space.nmo, space.nocc)
        if screening_space is None
        else screening_space
    )
    if (screen.nmo, screen.nocc) != (space.nmo, space.nocc):
        raise ValueError("Screening and excitation spaces must share a reference")
    state = build_static_screening(
        e, l, occupied=screen.occupied, virtual=screen.virtual, max_aux=cfg.max_aux
    )
    selected = jnp.asarray(space.occupied + space.virtual)
    valid = state.valid & jnp.all(jnp.isfinite(qp[selected]))
    for mask in (qp_computed_mask, qp_converged_mask):
        if mask is not None:
            mask = jnp.asarray(mask)
            if mask.shape != qp.shape or mask.dtype != jnp.bool_:
                raise ValueError(
                    "QP coverage/convergence masks must be boolean arrays of shape (nmo,)"
                )
            valid &= jnp.all(mask[selected])
    valid &= jnp.all(
        qp[jnp.asarray(space.virtual)][None, :]
        - qp[jnp.asarray(space.occupied)][:, None]
        > 0
    )
    if not cfg.tda:
        a, b = build_bse_operators(
            qp, l, space, state, singlet=cfg.singlet, block_size=cfg.block_size
        )
        solved = solve_rpa(
            a,
            b,
            config=EigenSolverConfig(
                method=cfg.solver,
                nroots=cfg.nroots,
                maxiter=cfg.max_cycle,
                max_subspace=cfg.max_space,
                atol=cfg.conv_tol,
                max_dense=cfg.max_dense,
                gradient_mode=cfg.gradient_mode,
                adjoint_tol=cfg.adjoint_tol,
                adjoint_maxiter=cfg.adjoint_max_cycle,
            ),
            gap_tol=cfg.gap_tol,
            seed=cfg.seed,
        )
        response_valid = solved.response_valid & valid
        energies = require_converged_derivative(solved.values, response_valid)
        shape = (cfg.nroots, len(space.occupied), len(space.virtual))
        x, y = [
            require_converged_derivative(
                v.T.reshape(shape), response_valid[:, None, None]
            )
            for v in (solved.x, solved.y)
        ]
        return BSEResult(
            energies,
            x,
            y,
            solved.residual_norms,
            solved.converged & valid,
            jnp.broadcast_to(solved.stable, energies.shape),
            response_valid,
            state.valid,
            state.min_gap,
            cfg.singlet,
            cfg.gradient_mode == "implicit_eigenvector",
            solved.stability_margins,
            solved.stability_certified,
            solved.stability_residual_norms,
        )
    op = build_tda_operator(
        qp, l, space, state, singlet=cfg.singlet, block_size=cfg.block_size
    )
    solved = solve_hermitian(
        op,
        config=EigenSolverConfig(
            method=cfg.solver,
            nroots=cfg.nroots,
            atol=cfg.conv_tol,
            maxiter=cfg.max_cycle,
            max_subspace=cfg.max_space,
            max_dense=cfg.max_dense,
            seed=cfg.seed,
        ),
        response=EigenResponseConfig(
            target=(
                "eigenpairs"
                if cfg.gradient_mode == "implicit_eigenvector"
                else "eigenvalues"
            ),
            gap_atol=cfg.gap_tol,
            gap_rtol=0.0,
            linear_config=LinearSolverConfig(
                rtol=cfg.adjoint_tol, maxiter=cfg.adjoint_max_cycle
            ),
        ),
    )
    energies = solved.values
    converged = solved.converged & valid
    stable = energies > cfg.gap_tol
    response_valid = solved.response_valid & valid & jnp.all(stable)
    energies = require_converged_derivative(energies, response_valid)
    x = solved.vectors[:, : cfg.nroots].T.reshape(
        cfg.nroots, len(space.occupied), len(space.virtual)
    )
    x = require_converged_derivative(x, response_valid[:, None, None])
    return BSEResult(
        energies,
        x,
        jnp.zeros_like(x),
        solved.residual_norms[: cfg.nroots],
        converged,
        stable,
        response_valid,
        state.valid,
        state.min_gap,
        cfg.singlet,
        cfg.gradient_mode == "implicit_eigenvector",
    )
