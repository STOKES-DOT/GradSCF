"""Shared attachment policy for molecular GW self-consistency response."""
import jax
import jax.numpy as jnp

from ..solvers import LinearSolverConfig


def require_real(*inputs):
    if any(jnp.iscomplexobj(x) for x in jax.tree.leaves(inputs)):
        raise NotImplementedError('Molecular GW outer response requires real inputs.')


def linear_config(config):
    if config.mode != 'implicit':
        raise ValueError("GW outer differentiation requires mode='implicit'.")
    if not config.require_converged or config.regularization != 0:
        raise ValueError('GW outer response requires converged states and zero regularization.')
    return LinearSolverConfig(rtol=config.tolerance, maxiter=config.max_iter,
                              restart=20 if config.restart is None else config.restart)


def require_valid(valid, residual, method):
    def fail(value):
        raise ArithmeticError(f'{method} outer solve failed convergence/response validity (residual {float(value):.3e}).')
    jax.lax.cond(valid, lambda: None, lambda: jax.debug.callback(fail, residual))
