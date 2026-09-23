"""Real nonnormal spectral response with biorthogonal left/right eigenvectors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def matrix():
    s = jnp.array([[1.0, 0.7, 0.2], [0.1, 1.0, -0.4], [0.2, 0.3, 1.0]])
    return s @ jnp.diag(jnp.array([0.5, 1.3, 2.1])) @ jnp.linalg.inv(s)


def test_nonnormal_pairs_and_energy_response():
    from gradscf.solvers import NonHermitianSolverConfig, solve_nonhermitian

    a = matrix()
    b = jnp.array([[0.2, 0.1, -0.1], [0.3, -0.2, 0.05], [0.1, -0.2, 0.3]])
    cfg = NonHermitianSolverConfig(nroots=2, atol=1e-11)
    out = jax.jit(lambda a: solve_nonhermitian(a, config=cfg))(a)
    assert np.all(out.converged & out.response_valid)
    np.testing.assert_allclose(out.values, [0.5, 1.3], atol=1e-12)
    np.testing.assert_allclose(
        out.left_vectors.T @ out.right_vectors, np.eye(2), atol=1e-12
    )
    np.testing.assert_allclose(
        a @ out.right_vectors, out.right_vectors * out.values, atol=1e-12
    )
    np.testing.assert_allclose(
        a.T @ out.left_vectors, out.left_vectors * out.values, atol=1e-12
    )
    f = lambda t: jnp.dot(
        solve_nonhermitian(a + t * b, config=cfg).values, jnp.array([0.3, 0.7])
    )
    ad = jax.jit(jax.grad(f))(0.0)
    np.testing.assert_allclose(ad, (f(1e-5) - f(-1e-5)) / 2e-5, atol=1e-9)
    np.testing.assert_allclose(jax.jvp(f, (0.0,), (1.0,))[1], ad, atol=1e-11)


@pytest.mark.parametrize(
    "a",
    [
        jnp.diag(jnp.array([1.0, 1.0, 3.0])),
        jnp.array([[1.0, 1.0], [0.0, 1.0]]),
        jnp.array([[0.0, -1.0], [1.0, 0.0]]),
    ],
)
def test_degenerate_defective_and_complex_response_is_invalid(a):
    from gradscf.solvers import NonHermitianSolverConfig, solve_nonhermitian

    cfg = NonHermitianSolverConfig(nroots=1)
    out = solve_nonhermitian(a, config=cfg)
    assert not np.any(out.response_valid)
    assert not np.isfinite(
        jax.grad(lambda t: solve_nonhermitian(a * t, config=cfg).values[0])(1.0)
    )
    assert not np.isfinite(
        jax.jvp(
            lambda t: solve_nonhermitian(a * t, config=cfg).values[0], (1.0,), (1.0,)
        )[1]
    )


def test_bounds_and_no_silent_symmetrization():
    from gradscf.solvers import NonHermitianSolverConfig, solve_nonhermitian

    a = matrix()
    assert (
        np.max(
            abs(
                np.linalg.eigvalsh(np.asarray((a + a.T) / 2))
                - np.array([0.5, 1.3, 2.1])
            )
        )
        > 0.01
    )
    with pytest.raises(ValueError, match="max_dense"):
        solve_nonhermitian(a, config=NonHermitianSolverConfig(max_dense=2))


def test_condition_limit_and_unselected_degeneracy():
    from gradscf.solvers import NonHermitianSolverConfig, solve_nonhermitian

    # A nonnormal but accurately solved eigenpair can have an invalid response.
    a = jnp.array([[1.0, 10.0], [0.0, 2.0]])
    out = solve_nonhermitian(a, config=NonHermitianSolverConfig(max_condition=2.0))
    assert np.all(out.converged)
    assert not np.any(out.response_valid)
    assert out.condition_numbers[0] > 10.0
    # Degenerate excluded states do not prevent the isolated first-root rule.
    a = jnp.diag(jnp.array([1.0, 2.0, 2.0]))
    f = lambda t: solve_nonhermitian(a + t * jnp.eye(3)).values[0]
    np.testing.assert_allclose(jax.grad(f)(0.0), 1.0, atol=1e-12)
    # Requesting the unresolved pair deliberately invalidates the whole set.
    cfg = NonHermitianSolverConfig(nroots=2)
    assert not np.any(solve_nonhermitian(a, config=cfg).response_valid)
