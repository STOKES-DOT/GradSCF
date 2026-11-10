"""Orbital-frame whitening without changing the symmetric inverse-root API."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import eigh


def orthogonalizer(overlap, eps=1e-10):
    from gradscf.solvers.scf import orbital_orthogonalizer
    return orbital_orthogonalizer(overlap, eps)


def problem(theta):
    factor = jnp.array([[1.1, .1+theta], [.2-theta*.3, .9]])
    overlap = factor @ factor.T + .4*jnp.eye(2)
    hcore = jnp.array([[-.8, .15], [.15, .6]])
    return overlap, hcore


def occupied_energy(theta):
    overlap, hcore = problem(theta)
    x = orthogonalizer(overlap)
    values = jnp.linalg.eigvalsh(x.T @ hcore @ x)
    return 2*values[0]


def test_real_spd_whitening_matches_generalized_eigen_oracle():
    overlap, hcore = problem(.13)
    x = jax.jit(orthogonalizer)(overlap)
    np.testing.assert_allclose(x.T @ overlap @ x, np.eye(2), atol=2e-13, rtol=0.)
    values, vectors = np.linalg.eigh(np.asarray(x.T @ hcore @ x))
    coeff = np.asarray(x) @ vectors
    expected, reference = eigh(np.asarray(hcore), np.asarray(overlap))
    np.testing.assert_allclose(values, expected, atol=2e-13, rtol=0.)
    np.testing.assert_allclose(coeff[:, :1] @ coeff[:, :1].T,
        reference[:, :1] @ reference[:, :1].T, atol=2e-13, rtol=0.)
    np.testing.assert_allclose(hcore @ coeff, overlap @ (coeff*values), atol=2e-13, rtol=0.)


@pytest.mark.parametrize("minimum", [0., .04])
def test_low_eigenvalue_clipping_preserves_existing_symmetric_factor(minimum):
    from gradscf.solvers.eigen.spectral import inverse_sqrt
    angle = .31
    rotation = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                          [jnp.sin(angle), jnp.cos(angle)]])
    overlap = (rotation*jnp.array([minimum, 1.7])) @ rotation.T
    actual = jax.jit(lambda s: orthogonalizer(s, .1))(overlap)
    expected = inverse_sqrt(overlap, .1)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0.)
    assert np.isfinite(np.asarray(actual)).all()


def test_clipped_branch_preserves_existing_directional_derivative():
    from gradscf.solvers.eigen.spectral import inverse_sqrt
    overlap = jnp.diag(jnp.array([.04, 1.7]))
    direction = jnp.array([[.01, .02], [.02, -.03]])
    actual = jax.jvp(lambda s: orthogonalizer(s, .1), (overlap,), (direction,))[1]
    expected = jax.jvp(lambda s: inverse_sqrt(s, .1), (overlap,), (direction,))[1]
    assert np.isfinite(np.asarray(actual)).all()
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0.)


def test_jvp_vjp_and_two_step_finite_differences_for_factor():
    overlap, _ = problem(.13)
    direction = jnp.array([[.2, -.11], [-.11, -.08]])
    weight = jnp.array([[.3, -.2], [.4, .1]])
    value, tangent = jax.jvp(orthogonalizer, (overlap,), (direction,))
    _, pullback = jax.vjp(orthogonalizer, overlap)
    np.testing.assert_allclose(jnp.vdot(weight, tangent),
        jnp.vdot(pullback(weight)[0], direction), atol=2e-12, rtol=0.)
    for step in (2e-4, 1e-4):
        finite = (orthogonalizer(overlap+step*direction)
                  - orthogonalizer(overlap-step*direction))/(2*step)
        np.testing.assert_allclose(tangent, finite, atol=2e-9, rtol=2e-7)
    assert np.isfinite(np.asarray(value)).all()


def test_occupied_energy_gradient_and_hvp_match_two_step_finite_differences():
    value_grad = jax.jit(jax.value_and_grad(occupied_energy))
    for theta in (.07, .13):
        value, gradient = value_grad(theta)
        hvp = jax.jvp(jax.grad(occupied_energy), (theta,), (.23,))[1]
        for step in (2e-4, 1e-4):
            finite_gradient = (occupied_energy(theta+step)-occupied_energy(theta-step))/(2*step)
            finite_hvp = .23*(value_grad(theta+step)[1]-value_grad(theta-step)[1])/(2*step)
            np.testing.assert_allclose(gradient, finite_gradient, atol=2e-8, rtol=2e-6)
            np.testing.assert_allclose(hvp, finite_hvp, atol=5e-8, rtol=2e-6)
        assert np.isfinite(float(value))


def test_repeated_overlap_eigenvalues_have_finite_first_and_second_response():
    perturbation = jnp.array([[.1, .07, -.03], [.07, -.2, .04], [-.03, .04, .15]])
    weight = jnp.arange(9, dtype=jnp.float64).reshape(3, 3)/11
    loss = lambda t: jnp.sum(orthogonalizer(jnp.eye(3)+t*perturbation)*weight)
    gradient = jax.jit(jax.grad(loss))(0.)
    hvp = jax.jvp(jax.grad(loss), (0.,), (.3,))[1]
    for step in (2e-4, 1e-4):
        fd_gradient = (loss(step)-loss(-step))/(2*step)
        fd_hvp = .3*(jax.grad(loss)(step)-jax.grad(loss)(-step))/(2*step)
        np.testing.assert_allclose(gradient, fd_gradient, atol=2e-9, rtol=2e-7)
        np.testing.assert_allclose(hvp, fd_hvp, atol=2e-9, rtol=2e-7)
    assert np.isfinite(np.asarray([gradient, hvp])).all()


def test_symmetric_inverse_root_retains_its_original_contract():
    from gradscf.solvers.eigen.spectral import inverse_sqrt
    overlap, _ = problem(.13)
    symmetric = inverse_sqrt(overlap, 1e-10)
    np.testing.assert_allclose(symmetric, symmetric.T, atol=1e-14, rtol=0.)
    np.testing.assert_allclose(symmetric @ symmetric @ overlap, np.eye(2), atol=2e-13, rtol=0.)
