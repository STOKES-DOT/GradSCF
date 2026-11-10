"""Checked eager fixed-point iteration for maps with host domain validation."""
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .diis import diis_push, diis_solve
from .iterate import NonlinearConfig


class EagerIterationResult(NamedTuple):
    solution: object
    iterations: int
    residual_norm: float
    converged: bool


def iterate_eager(physical_map, initial, *, config=None, valid_state=None,
                  check=None, callback=None):
    """Iterate a packed real vector using shared DIIS and true map residuals.

    Convergence requires max(abs(map(x)-x)) <= residual_tol at the returned
    state. Optional check(x,mapped,norm) can only add a convergence condition;
    callback(iteration,x,norm) observes it without changing that condition.
    NonlinearConfig.energy_tol is left to check because no physical energy is
    defined by a generic packed vector. diis_space=1 selects plain iteration.
    Invalid DIIS proposals use the damped plain map, then at most 16 halvings
    toward the current valid state. No expensive map is evaluated outside
    valid_state. This forward solver explicitly supplies no AD/JIT contract.
    """
    legacy_clean = getattr(jax.core, 'trace_state_clean', None)
    if legacy_clean is None:
        from jax.extend import core
        trace_state = getattr(core, 'get_opaque_trace_state', None) or jax.core.get_opaque_trace_state
        take_trace = getattr(core, 'take_current_trace', None) or jax.core.take_current_trace
        active_trace = trace_state()
        with take_trace():
            eager_trace = trace_state()
        eager = active_trace == eager_trace
    else:
        eager = legacy_clean()
    if not eager or any(isinstance(x, jax.core.Tracer)
            for x in jax.tree_util.tree_leaves(initial)):
        raise NotImplementedError('iterate_eager is eager forward only; AD/JIT is unsupported')
    cfg = NonlinearConfig() if config is None else config
    state = jnp.asarray(initial)
    if state.ndim != 1 or not jnp.issubdtype(state.dtype, jnp.floating):
        raise ValueError('Use a packed real floating-point vector')
    def valid(value):
        return bool(jnp.all(jnp.isfinite(value))) and (valid_state is None or bool(valid_state(value)))
    if not valid(state):
        raise ValueError('The initial eager state is outside its finite physical domain')
    history = jnp.zeros((cfg.diis_space, state.size), dtype=state.dtype)
    errors = jnp.zeros_like(history)
    head, count = jnp.array(0), jnp.array(0)
    for iteration in range(cfg.maxiter+1):
        mapped = jnp.asarray(physical_map(state))
        if mapped.shape != state.shape or not jnp.issubdtype(mapped.dtype, jnp.floating):
            raise ValueError('The physical map must return a matching real floating-point vector')
        residual = mapped-state
        norm = float(jnp.max(jnp.abs(residual))) if state.size else 0.
        if callback is not None:
            callback(iteration, state, norm)
        done = norm <= cfg.residual_tol and (check is None or bool(check(state, mapped, norm)))
        if done or iteration == cfg.maxiter or not isfinite(norm):
            return EagerIterationResult(state, iteration, norm, done)
        plain = state+(1-cfg.damping)*residual
        history, errors, head, count = diis_push(plain, residual, history, errors, head, count)
        candidate = (diis_solve(history, errors, count)
                     if int(count) >= 2 and iteration+1 >= cfg.diis_start else plain)
        if not valid(candidate):
            candidate = plain
            for _ in range(16):
                if valid(candidate):
                    break
                candidate = .5*(candidate+state)
            if not valid(candidate):
                return EagerIterationResult(state, iteration, norm, False)
        state = candidate


__all__ = ['EagerIterationResult', 'iterate_eager']


def minimize_eager(value_and_grad, initial, *, maxiter=300, gradient_tolerance=1e-8,
                   callback=None, hessian=None):
    """Checked L-BFGS with optional AD-Jacobian stationarity refinement; no inverse-map AD/JIT contract.

    The callback returns a scalar and its real vector gradient. Keep physical
    objectives and JAX compilation in the caller; this shared driver owns only
    host optimization and rejects invalid objective/gradient evaluations.
    """
    import numpy as np
    from scipy.optimize import minimize, root

    if any(isinstance(a, jax.core.Tracer) for a in jax.tree.leaves(initial)):
        raise NotImplementedError('minimize_eager is offline; inverse-map AD/JIT is unsupported')
    x = np.asarray(initial)
    if x.ndim != 1 or np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError('Use a finite real vector as the initial state')
    if not isinstance(maxiter, int) or maxiter < 1 or not isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError('Positive maxiter and gradient_tolerance are required')

    def objective(vector):
        value, gradient = value_and_grad(jnp.asarray(vector))
        value, gradient = np.asarray(value), np.asarray(gradient)
        if value.shape != () or gradient.shape != vector.shape or np.iscomplexobj(value) or np.iscomplexobj(gradient):
            raise ValueError('The objective requires a real scalar and matching real gradient')
        if not np.isfinite(value).all() or not np.isfinite(gradient).all():
            raise ValueError('The objective and gradient must be finite')
        return float(value), np.asarray(gradient, dtype=float)

    reserve = min(100, maxiter//4) if hessian is not None else 0
    result = minimize(objective, np.asarray(x, dtype=float), jac=True, method='L-BFGS-B',
                      callback=callback, options={'maxiter': maxiter-reserve, 'gtol': gradient_tolerance,
                      'ftol': 1e-15, 'maxls': 50, 'maxcor': 20})
    result['method'] = 'L-BFGS-B'
    # Preserve the stable quasi-Newton trajectory on flat/semi-definite
    # objectives. Actual curvature is used only to polish unresolved residuals.
    if hessian is not None and np.max(np.abs(result.jac)) > gradient_tolerance and result.nit < maxiter:
        def curvature(vector):
            value = np.asarray(hessian(jnp.asarray(vector)))
            if value.shape != (vector.size, vector.size) or np.iscomplexobj(value) or not np.isfinite(value).all():
                raise ValueError('The Hessian must be a finite real square matrix')
            return .5*(value+value.T)
        # Near stationarity, an energy decrease can be smaller than one ULP.
        # Solve the gradient residual with its actual AD Jacobian instead of
        # accepting a false energy-change convergence certificate.
        refined = root(lambda vector: (objective(vector)[1], curvature(vector)), result.x,
                       jac=True, method='lm', options={'maxiter': maxiter-result.nit,
                       'ftol': 1e-12, 'xtol': 1e-12, 'gtol': 1e-12})
        refined_value, refined_gradient = objective(refined.x)
        allowance = 64*np.finfo(x.dtype).eps*max(1., abs(result.fun))
        result['refinement_evaluations'] = int(refined.nfev)
        result['nfev'] += int(refined.nfev)+1
        result['njev'] += int(getattr(refined, 'njev', 0))
        if np.max(np.abs(refined_gradient)) < np.max(np.abs(result.jac)) and refined_value <= result.fun+allowance:
            result.update(x=refined.x, fun=refined_value, jac=refined_gradient,
                          success=bool(np.max(np.abs(refined_gradient)) <= gradient_tolerance),
                          message='Quasi-Newton objective followed by checked stationarity refinement',
                          method='L-BFGS-B+stationary-lm')
    return result


__all__ += ['minimize_eager']
