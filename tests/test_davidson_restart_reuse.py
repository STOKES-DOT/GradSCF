"""Coverage for fixed-H restart reuse and original-operator Ritz diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import EigenSolverConfig, solve_hermitian
from gradscf.solvers.eigen import davidson


def _restarting_problem():
    rng = np.random.default_rng(9)
    array = rng.normal(size=(40, 40))
    matrix = np.diag(np.linspace(0., 10., len(array))) + .15 * (array + array.T)
    config = EigenSolverConfig(
        nroots=1, atol=1e-10, maxiter=100, max_subspace=8, collapse_subspace=3,
    )
    return jnp.asarray(matrix), config


@pytest.mark.parametrize("force_fresh", [False, True])
def test_fixed_operator_restart_counts_reuse_and_unsafe_fallback(monkeypatch, force_fresh):
    if force_fresh:
        monkeypatch.setattr(davidson, "_restart_qr_is_stable", lambda r: jnp.asarray(False))
    matrix, config = _restarting_problem()
    events = []

    solve = jax.jit(lambda a: solve_hermitian(a, config=config, workload_callback=events.append))
    result = jax.block_until_ready(solve(matrix))
    jax.effects_barrier()

    restart = [event for event in events if event["purpose"] == "restart"]
    assert restart, "Fixture must reach a thick restart"
    if force_fresh:
        assert any(event["event"] == "apply_end" and event["columns"] == 3 for event in restart)
        assert all(event["event"] != "reuse" for event in restart)
    else:
        assert all(event["event"] == "reuse" and event["effective_columns"] == 0 for event in restart)

    vectors = np.asarray(result.raw_vectors)
    values = np.asarray(result.raw_values)
    residual = np.asarray(matrix) @ vectors - vectors * values
    expected = np.linalg.eigvalsh(matrix)[:2]
    np.testing.assert_allclose(values, expected, rtol=0., atol=1e-10)
    assert np.linalg.norm(residual, axis=0).max() <= config.atol
    assert bool(result.response_valid[0])
    assert sum(event["purpose"] == "final_ritz_verify" and event["event"] == "apply_end"
               for event in events) == 1


def test_restart_rank_guard_checks_offdiagonal_as_well_as_diagonal():
    assert bool(davidson._restart_qr_is_stable(jnp.eye(3)))
    assert not bool(davidson._restart_qr_is_stable(jnp.diag(jnp.asarray([1., 0., 1.]))))
    assert not bool(davidson._restart_qr_is_stable(jnp.asarray([[1., 1.e8], [0., 1.]])))
    assert not bool(davidson._restart_qr_is_stable(jnp.diag(jnp.asarray([1., 1.e-4]))))


def test_verified_infinity_norm_uses_fresh_ritz_residual():
    matrix, config = _restarting_problem()
    result, inf_norms = jax.jit(
        lambda a: solve_hermitian(a, config=config, _return_residual_inf=True)
    )(matrix)
    vectors = np.asarray(result.raw_vectors)
    values = np.asarray(result.raw_values)
    residual = np.asarray(matrix) @ vectors - vectors * values
    np.testing.assert_allclose(inf_norms, np.max(np.abs(residual), axis=0), rtol=0., atol=5e-15)
    np.testing.assert_allclose(result.raw_residual_norms, np.linalg.norm(residual, axis=0),
                               rtol=0., atol=5e-15)
