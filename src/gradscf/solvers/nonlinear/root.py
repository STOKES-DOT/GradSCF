"""Implicit differentiation of a converged root supplied by a caller."""
import jax
from ..linear.implicit import checked_linear_solve
from ..types import LinearSolverConfig


def attach_root(residual, solution, *, config=None, tangent_solve=None, converged=True):
    """Attach root JVP/VJP and higher response without differentiating the trajectory.

    residual may close over differentiable parameters. Numerical forward
    iteration and the physical convergence criterion are supplied by the caller.
    An optional tangent_solve supports structured response operators.
    """
    cfg = LinearSolverConfig() if config is None else config
    seed = jax.lax.stop_gradient(solution)
    if tangent_solve is None:
        def tangent_solve(operator, rhs):
            shape = rhs.shape
            return checked_linear_solve(
                lambda v: operator(v.reshape(shape)).reshape(-1), rhs.reshape(-1),
                config=cfg, converged=converged)[0].reshape(shape)
    return jax.lax.custom_root(residual, seed,
        solve=lambda _residual, _initial: seed, tangent_solve=tangent_solve)
