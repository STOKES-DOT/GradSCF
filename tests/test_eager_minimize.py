"""Shared eager minimization consumes a checked real value/gradient callback."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_eager_minimize_quadratic():
    from gradscf.solvers.nonlinear import minimize_eager
    objective = jax.jit(jax.value_and_grad(lambda x: jnp.sum(jnp.array([2., 5.])*(x-jnp.array([.3, -.7]))**2)))
    result = minimize_eager(objective, jnp.zeros(2), gradient_tolerance=1e-10)
    assert result.success
    np.testing.assert_allclose(result.x, [.3, -.7], atol=1e-8)


def test_eager_minimize_rejects_nonfinite_gradient():
    from gradscf.solvers.nonlinear import minimize_eager
    with pytest.raises(ValueError, match='finite'):
        minimize_eager(lambda x: (jnp.sum(x*x), jnp.full_like(x, jnp.nan)), jnp.zeros(2))


def test_eager_minimize_uses_supplied_hessian_for_strict_stationarity():
    from gradscf.solvers.nonlinear import minimize_eager
    center = jnp.array([.3, -.7])
    def objective(x):
        delta = x-center
        return 1e6+jnp.sum(jnp.array([1e-4, 100.])*delta**2)
    result = minimize_eager(jax.jit(jax.value_and_grad(objective)), jnp.zeros(2),
                            hessian=jax.jit(jax.hessian(objective)), gradient_tolerance=1e-10)
    assert np.max(np.abs(result.jac)) <= 1e-10
    np.testing.assert_allclose(result.x, center, atol=1e-8)


def test_stationarity_refinement_does_not_require_resolving_sub_ulp_energy_changes():
    from gradscf.solvers.nonlinear import minimize_eager
    center = jnp.array([.3, -.7])
    def objective(x):
        return 1e6+.5*jnp.sum(jnp.array([1.,3.])*(x-center)**2)
    result = minimize_eager(jax.jit(jax.value_and_grad(objective)), center+jnp.array([3e-9,-2e-9]),
                            hessian=jax.jit(jax.hessian(objective)), gradient_tolerance=1e-11)
    assert np.max(np.abs(result.jac)) <= 1e-11


def test_curvature_refinement_has_a_reserved_budget_after_slow_quasi_newton_steps():
    from gradscf.solvers.nonlinear import minimize_eager
    center = jnp.linspace(.2, 1.2, 6)
    weights = 10.**jnp.arange(-4., 2.)
    objective = lambda x: .5*jnp.sum(weights*(x-center)**2)
    result = minimize_eager(jax.jit(jax.value_and_grad(objective)), jnp.zeros(6),
                            hessian=jax.jit(jax.hessian(objective)), maxiter=12,
                            gradient_tolerance=1e-10)
    assert np.max(np.abs(result.jac)) <= 1e-10
