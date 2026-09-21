"""Restricted left-state equations as a transposed residual Jacobian action."""

import jax
import jax.numpy as jnp
from ..solvers import LinearOperator, solve_linear
from .types import CCConfig, LambdaResult
from .ground import METHODS, linear_config
from .integrals import prepare_integrals
from .amplitudes import AmplitudeSpace
from .rccsd import residual, correlation_energy


def solve_lambda(h1, eri, result, *, nocc, frozen=None, config=None):
    """Solve J^T lambda = -dE/dt in independent norm-preserving coordinates.

    Convert the packed dual to the conventional restricted spin-adapted l1/l2:
    the full-tensor pairing is 2*l1*R1 + (2*l2-l2.swap(a,b))*R2.
    Lambda convergence is distinct from convergence of the right amplitudes.
    """
    cfg = CCConfig() if config is None else config
    ints = prepare_integrals(h1, eri, nocc=nocc, frozen=frozen)
    space = AmplitudeSpace(ints.nocc, ints.nvir, cfg.method)
    x = space.pack(result.t1, result.t2)
    r = lambda t: space.pack(*residual(*space.unpack(t), ints, model=cfg.method))
    e = lambda t: correlation_energy(*space.unpack(t), ints, model=cfg.method)
    current, pullback = jax.vjp(r, x)
    valid = (
        result.converged
        & (result.method_id == METHODS.index(cfg.method))
        & (jnp.max(jnp.abs(current), initial=0.0) <= cfg.residual_tol * 10)
    )
    rhs = jnp.where(valid, -jax.grad(e)(x), jnp.nan)
    op = LinearOperator((x.size, x.size), x.dtype, lambda v: pullback(v)[0])
    solved = solve_linear(op, rhs, config=linear_config(cfg))
    dual1, dual2 = space.unpack(solved.solution)
    l1 = dual1 / 2
    l2 = (2 * dual2 + dual2.swapaxes(2, 3)) / 3
    return LambdaResult(
        l1, l2, solved.solution, solved.residual_norm, solved.converged & valid
    )
