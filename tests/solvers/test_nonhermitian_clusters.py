"""Roundoff complex representations must not destroy a real eigenspace."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.solvers import NonHermitianSolverConfig, solve_nonhermitian


def repeated_matrix(seed=4):
    rng = np.random.default_rng(seed)
    s = np.eye(5) + 0.3 * rng.normal(size=(5, 5))
    return jnp.asarray(s @ np.diag([0.5, 0.5, 1.3, 2.1, 3.0]) @ np.linalg.inv(s))


@pytest.mark.parametrize("method", ["dense", "davidson"])
@pytest.mark.parametrize("nroots", [1, 2, 3])
def test_real_repeated_root_keeps_independent_dual_bases(method, nroots):
    a = repeated_matrix()
    cfg = NonHermitianSolverConfig(method=method, nroots=nroots, max_space=12)
    out = jax.jit(lambda a: solve_nonhermitian(a, config=cfg))(a)
    assert np.all(out.converged)
    assert not np.any(out.response_valid)
    np.testing.assert_allclose(out.values, [0.5, 0.5, 1.3][:nroots], atol=1e-12)
    np.testing.assert_allclose(
        out.left_vectors.T @ out.right_vectors, np.eye(nroots), atol=1e-11
    )
    np.testing.assert_allclose(
        a @ out.right_vectors, out.right_vectors * out.values, atol=1e-11
    )
    np.testing.assert_allclose(
        a.T @ out.left_vectors, out.left_vectors * out.values, atol=1e-11
    )
    assert not np.isfinite(
        jax.grad(lambda t: solve_nonhermitian(a * t, config=cfg).values.sum())(1.0)
    )


def test_excluded_defect_does_not_invalidate_an_isolated_root():
    a = jnp.array([[0.2, 0.1, -0.2], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    out = solve_nonhermitian(a)
    assert out.converged[0] and out.response_valid[0]
    np.testing.assert_allclose(
        jax.grad(lambda t: solve_nonhermitian(a * t).values[0])(1.0), 0.2, atol=1e-12
    )


@pytest.mark.parametrize("method", ["dense", "davidson"])
def test_defective_repeated_cluster_is_not_repaired_into_valid_eigenpairs(method):
    a = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    cfg = NonHermitianSolverConfig(method=method, nroots=2, max_space=8, maxiter=4)
    out = solve_nonhermitian(a, config=cfg)
    assert not np.all(out.converged)
    assert not np.any(out.response_valid)


def test_nearby_distinct_roots_keep_their_eigenvectors():
    s = jnp.array([[1.0, 0.3, 0.1], [0.1, 1.0, 0.2], [0.0, 0.2, 1.0]])
    a = s @ jnp.diag(jnp.array([0.5, 0.500001, 2.0])) @ jnp.linalg.inv(s)
    cfg = NonHermitianSolverConfig(nroots=2, atol=1e-11)
    out = solve_nonhermitian(a, config=cfg)
    assert np.all(out.converged & out.response_valid)
    np.testing.assert_allclose(
        a @ out.right_vectors, out.values * out.right_vectors, atol=1e-11
    )
