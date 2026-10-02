"""Reuse a real symmetric factorization without eigenvector derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.solvers.linear.shifted import factor_shifted, solve_shifted


@pytest.mark.parametrize('shift', [.2, .2+.1j])
def test_degenerate_shifted_solve_response(shift):
    matrix = jnp.eye(3) * 1.7
    direction = jnp.array([[.2, .1, -.3], [.1, -.4, .2], [-.3, .2, .1]])
    rhs = jnp.arange(6., dtype=float).reshape(3, 2) / 5
    def solve(t, reference=False):
        a = matrix + t * direction
        b = rhs * (1+.1*t)
        z = shift + .03*t
        return (jnp.linalg.solve(a-z*jnp.eye(3), b) if reference else
                solve_shifted(factor_shifted(a), b, z))
    actual = jax.jit(solve)(0.)
    np.testing.assert_allclose(actual, solve(0., True), atol=2e-13)
    tangent = jax.jit(lambda t: jax.jvp(solve, (t,), (1.,))[1])(0.)
    expected = jax.jvp(lambda t: solve(t, True), (0.,), (1.,))[1]
    np.testing.assert_allclose(tangent, expected, atol=2e-12)
    loss = lambda t: jnp.real(jnp.vdot(solve(t), solve(t)))
    expected_loss = lambda t: jnp.real(jnp.vdot(solve(t, True), solve(t, True)))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), jax.grad(expected_loss)(0.), atol=2e-12)
    np.testing.assert_allclose(jax.grad(loss)(0.), (loss(1e-5)-loss(-1e-5))/2e-5, atol=2e-9)


def test_degenerate_basis_rotation_and_multiple_shifts():
    matrix = jnp.eye(3)*2
    state = factor_shifted(matrix)
    q, _ = jnp.linalg.qr(jnp.array([[.1, .3, .7], [.8, .4, .2], [.6, .9, .5]]))
    rotated = state._replace(vectors=state.vectors @ q)
    rhs = jnp.array([1., -.2, .5])
    values = jax.jit(jax.vmap(lambda z: solve_shifted(rotated, rhs, z)))(jnp.array([.1, .5, 1.]))
    np.testing.assert_allclose(values, rhs[None, :]/(2-jnp.array([.1, .5, 1.]))[:,None], atol=2e-13)


@pytest.mark.parametrize('defective', [False, True])
def test_complex_symmetric_shifted_response_and_defective_fallback(defective):
    # The second matrix has a nonzero nilpotent block and no complete eigenbasis.
    matrix = (jnp.array([[1j, 1.], [1., -1j]])+2*jnp.eye(2) if defective else
              jnp.array([[1.7+.2j, .13-.02j], [.13-.02j, 2.1-.1j]]))
    direction = jnp.array([[.02+.01j, -.03j], [-.03j, -.04+.02j]])
    rhs = jnp.array([.2+.1j, .7-.3j])
    if defective:
        assert not bool(factor_shifted(matrix).spectral_valid)
    def solve(t, reference=False):
        a, b, z = matrix+t*direction, rhs*(1+.03*t), .2+.1j+.02*t
        return (jnp.linalg.solve(a-z*jnp.eye(2), b) if reference else
                solve_shifted(factor_shifted(a), b, z))
    np.testing.assert_allclose(jax.jit(solve)(0.), solve(0., True), atol=2e-12)
    np.testing.assert_allclose(jax.jvp(solve, (0.,), (1.,))[1],
        jax.jvp(lambda t: solve(t, True), (0.,), (1.,))[1], atol=2e-12)
    loss = lambda t: jnp.real(jnp.vdot(solve(t), solve(t)))
    exact = lambda t: jnp.real(jnp.vdot(solve(t, True), solve(t, True)))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), jax.grad(exact)(0.), atol=2e-12)
