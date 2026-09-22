"""Tensor Kronecker-sum inverse: dense oracle and repeated-eigenvalue AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_tensor_sum_solution_and_degenerate_response():
    from gradscf.solvers import solve_tensor_sum
    a = jnp.diag(jnp.array([2., 2., 3.]))
    b = jnp.diag(jnp.array([.7, 1.1]))
    rhs = jnp.asarray(np.random.default_rng(41).normal(size=(3, 2)))
    da = jnp.array([[.1, .3, -.2], [.3, -.1, .07], [-.2, .07, .2]])
    db = jnp.array([[.05, -.12], [-.12, -.03]])
    weight = jnp.asarray(np.random.default_rng(42).normal(size=rhs.shape))
    def value(x):
        return jnp.sum(weight*solve_tensor_sum((a+x*da, b+x*db), rhs).solution)
    out = jax.jit(lambda a, b: solve_tensor_sum((a, b), rhs))(a, b)
    full = np.kron(a, np.eye(2))+np.kron(np.eye(3), b)
    np.testing.assert_allclose(out.solution.ravel(), np.linalg.solve(full, rhs.ravel()), atol=1e-13)
    assert out.converged
    derivative = jax.jit(jax.grad(value))(0.)
    np.testing.assert_allclose(derivative, (value(1e-4)-value(-1e-4))/2e-4, atol=2e-9, rtol=0)
    # The independent dense resolvent establishes response at exact degeneracy.
    dense = lambda x: jnp.vdot(weight.ravel(), jnp.linalg.solve(
        jnp.kron(a+x*da, jnp.eye(2))+jnp.kron(jnp.eye(3), b+x*db), rhs.ravel()))
    np.testing.assert_allclose(derivative, jax.grad(dense)(0.), atol=1e-13)
    np.testing.assert_allclose(jax.jvp(value, (0.,), (1.,))[1], derivative, atol=1e-13)


def test_tensor_sum_failure_and_shapes():
    from gradscf.solvers import solve_tensor_sum
    a, b = jnp.eye(2), -jnp.eye(2)
    out = solve_tensor_sum((a, b), jnp.ones((2, 2)))
    assert not out.converged and np.isnan(out.solution).all()
    assert not np.isfinite(jax.grad(lambda x: jnp.sum(
        solve_tensor_sum((a*x, b), jnp.ones((2, 2))).solution))(1.))
    with pytest.raises(ValueError, match="shape"):
        solve_tensor_sum((a, a), jnp.zeros((3, 2)))
    with pytest.raises(ValueError, match="floating"):
        solve_tensor_sum((a,), jnp.ones(2, dtype=jnp.complex128))
    nonsymmetric = a.at[0, 1].set(.1)
    assert not solve_tensor_sum((nonsymmetric, a), jnp.ones((2, 2))).converged


def test_empty_tensor_validates_remaining_factors():
    from gradscf.solvers import solve_tensor_sum
    empty = jnp.empty((0, 0))
    rhs = jnp.empty((0, 2))
    assert solve_tensor_sum((empty, jnp.eye(2)), rhs).converged
    assert not solve_tensor_sum((empty, jnp.full((2, 2), jnp.nan)), rhs).converged
    assert not solve_tensor_sum((empty, jnp.array([[1., .2], [0., 1.]])), rhs).converged
