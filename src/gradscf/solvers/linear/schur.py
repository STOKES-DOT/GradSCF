"""One-scalar bordered systems with checked block and transpose solves."""
import jax
import jax.numpy as jnp
from ..types import LinearSolverConfig
from ..diagnostics import linear_residual
from .implicit import checked_linear_solve


def solve_scalar_border(matvec, rhs, *, config=None, converged=True, schur_valid=None):
    """Solve [[A,b],[c,d]] x = rhs using a scalar Schur complement.

    A caller can supply schur_valid(schur) for a physical resolution threshold.
    Generic cancellation and true-residual checks apply in both orientations.
    """
    cfg = LinearSolverConfig() if config is None else config
    if rhs.ndim != 1 or rhs.size < 1:
        raise ValueError("A scalar-bordered RHS must be a nonempty vector")
    def checked_solve(operator, value):
        zero = jnp.zeros(1, dtype=value.dtype)
        lift = lambda v: jnp.concatenate((v, zero))
        block = lambda v: operator(lift(v))[:-1]
        column = operator(jnp.zeros_like(value).at[-1].set(1.0))
        def solve(v):
            return checked_linear_solve(block, v, config=cfg, converged=converged)[0]
        response_column = solve(column[:-1])
        response_rhs = solve(value[:-1])
        eliminated = operator(lift(response_column))[-1]
        schur = column[-1] - eliminated
        cancellation_floor = cfg.rtol * (jnp.abs(column[-1]) + jnp.abs(eliminated))
        resolved = jnp.abs(schur) > cancellation_floor
        if schur_valid is not None:
            resolved = resolved & schur_valid(schur)
        last = (value[-1] - operator(lift(response_rhs))[-1]) / jnp.where(resolved, schur, 1.)
        solution = jnp.concatenate((response_rhs - response_column*last, last[None]))
        _, valid = linear_residual(operator, solution, value, rtol=cfg.rtol, atol=cfg.atol,
                                    converged=converged & resolved)
        return jnp.where(valid, solution, jnp.full_like(solution, jnp.nan))
    return jax.lax.custom_linear_solve(matvec, rhs, solve=checked_solve,
                                       transpose_solve=checked_solve)
