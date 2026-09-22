"""Static physical problem assembly through GradSCF's common eigensolver."""

import numpy as np
import jax
import jax.numpy as jnp
from ..solvers import EigenSolverConfig, solve_hermitian
from ..solvers.diagnostics import require_converged_derivative
from ..gw.screened import build_static_screening
from .space import make_bse_space
from .types import BSEConfig, BSEResult
from .kernel import build_tda_operator


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
    """Real static TDA-BSE with explicit QP and screening spectra.

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
    op = build_tda_operator(
        qp, l, space, state, singlet=cfg.singlet, block_size=cfg.block_size
    )
    nsolve = min(cfg.nroots + 1, space.size)
    width = min(cfg.max_space, space.size)
    if cfg.solver == "davidson" and width < nsolve:
        raise ValueError(
            "max_space must include requested roots and the BSE guard root"
        )
    initial = None
    if cfg.solver == "davidson":
        k = min(width, max(nsolve, 2 * nsolve))
        initial = (
            jnp.asarray(np.random.default_rng(cfg.seed).normal(size=(space.size, k)))
            * 0.05
        )
        initial = initial.at[jnp.argsort(op.diagonal)[:k], jnp.arange(k)].add(1.0)
    solved = solve_hermitian(
        op,
        config=EigenSolverConfig(
            method=cfg.solver,
            nroots=nsolve,
            atol=cfg.conv_tol,
            maxiter=cfg.max_cycle,
            max_subspace=width,
            max_dense=cfg.max_dense,
            gradient_mode=cfg.gradient_mode,
            adjoint_tol=cfg.adjoint_tol,
            adjoint_maxiter=cfg.adjoint_max_cycle,
        ),
        initial_vectors=initial,
    )
    energies = solved.values[: cfg.nroots]
    distances = jnp.abs(energies[:, None] - solved.values[None, :])
    distances = distances.at[jnp.arange(cfg.nroots), jnp.arange(cfg.nroots)].set(
        jnp.inf
    )
    margin = (
        cfg.gap_tol
        + solved.residual_norms[: cfg.nroots, None]
        + solved.residual_norms[None, :]
    )
    isolated = jnp.all(distances > margin, axis=1) & jnp.all(solved.converged)
    converged = solved.converged[: cfg.nroots] & valid
    stable = energies > cfg.gap_tol
    response_valid = converged & stable & isolated
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
