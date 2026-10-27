"""Opt-in runtime workload accounting for the bounded Davidson solvers."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.solvers.eigen.davidson import _davidson_lowest_symmetric
from gradscf.solvers.eigen.rpa import (
    _davidson_lowest_tdhf,
    implicit_differential_davidson_lowest_tdhf,
)


def _matrix():
    # One requested root is already exact; only the second needs expansion.
    diagonal = jnp.arange(8, dtype=jnp.float64) + 0.5
    coupling = jnp.asarray([0.0] + [0.12] * 6, dtype=diagonal.dtype)
    return jnp.diag(diagonal) + jnp.diag(coupling, 1) + jnp.diag(coupling, -1)


def _rpa_operator(a, b):
    def vind(rows):
        x, y = jnp.split(rows, 2, axis=-1)
        return jnp.concatenate([x @ a.T + y @ b.T, -(x @ b.T + y @ a.T)], axis=-1)

    return vind


def _ready(result):
    jax.block_until_ready(result)
    jax.effects_barrier()
    return result


def test_symmetric_workload_reports_padded_actions_and_root_status():
    matrix = _matrix()
    events = []
    options = dict(nroots=2, tol=1e-9, max_iter=10, max_subspace=8)
    baseline = _ready(_davidson_lowest_symmetric(matrix, **options))
    observed = _ready(_davidson_lowest_symmetric(
        matrix, workload_callback=events.append, **options,
    ))

    for reference, value in zip(baseline, observed):
        np.testing.assert_array_equal(value, reference)
    assert bool(observed[2])
    np.testing.assert_allclose(observed[0], np.linalg.eigvalsh(matrix)[:2], atol=1e-9)
    initial = next(event for event in events
                   if event["event"] == "apply_end" and event["purpose"] == "initial")
    assert initial["columns"] == initial["effective_columns"] == 2
    append = next(event for event in events
                  if event["event"] == "apply_end" and event["purpose"] == "append")
    assert append["columns"] == 2
    assert append["effective_columns"] == 1
    first_iteration = next(event for event in events if event["event"] == "iteration")
    assert first_iteration["iteration"] == 1
    assert first_iteration["subspace_dim"] == 2
    assert first_iteration["active_roots"] == 1
    assert first_iteration["converged_roots"] == 1
    assert first_iteration["locked_roots"] == 0
    assert len(first_iteration["residual_norms"]) == 2
    assert all(event["elapsed_seconds"] >= 0 for event in events)
    assert all(event["timing_kind"] == "ordered_host_callbacks" for event in events)
    assert [(event["iteration"], event["purpose"]) for event in events
            if event["event"] == "apply_start"] == [
        (event["iteration"], event["purpose"]) for event in events
        if event["event"] == "apply_end"
    ]


def test_rpa_workload_records_true_residual_actions_without_changing_result():
    a = _matrix()
    b = jnp.eye(a.shape[0], dtype=a.dtype) * 0.03
    vind = _rpa_operator(a, b)
    options = dict(nroots=2, size=8, diag=jnp.diag(a), tol=1e-8,
                   max_iter=20, max_subspace=8)
    baseline = _ready(_davidson_lowest_tdhf(vind, **options))
    events = []
    observed = _ready(_davidson_lowest_tdhf(
        vind, workload_callback=events.append, **options,
    ))
    for reference, value in zip(baseline, observed):
        np.testing.assert_array_equal(value, reference)
    assert bool(observed[3])
    residual_actions = [event for event in events
                        if event["event"] == "apply_end" and event["purpose"] == "residual"]
    assert residual_actions
    assert all(event["columns"] == 2 for event in residual_actions)
    assert all(len(event["residual_norms"]) == 2 for event in residual_actions)
    assert any(event["converged_roots"] == 1 for event in residual_actions)
    assert all(event["locked_roots"] == 0 for event in events)


def test_workload_callback_is_absent_from_clean_jaxpr_and_runs_under_jit():
    matrix = _matrix()
    options = dict(nroots=2, tol=1e-8, max_iter=10, max_subspace=8)
    solve = lambda value: _davidson_lowest_symmetric(value, **options)
    assert "debug_callback" not in str(jax.make_jaxpr(solve)(matrix))
    events = []
    traced = jax.jit(lambda value: _davidson_lowest_symmetric(
        value, workload_callback=events.append, **options,
    ))
    result = _ready(traced(matrix))
    assert bool(result[2])
    assert any(event["event"] == "iteration" for event in events)


def test_rpa_workload_preserves_implicit_eigenvalue_jvp():
    a = _matrix()
    b = jnp.eye(a.shape[0], dtype=a.dtype) * 0.03
    direction = jnp.eye(a.shape[0], dtype=a.dtype) * 0.07
    options = dict(nroots=2, size=8, diag=jnp.diag(a), tol=1e-8,
                   max_iter=20, max_subspace=8)
    events = []

    def solve(matrix, callback):
        return implicit_differential_davidson_lowest_tdhf(
            _rpa_operator(matrix, b), workload_callback=callback, **options,
        )[0]

    baseline = _ready(jax.jvp(lambda matrix: solve(matrix, None), (a,), (direction,)))
    observed = _ready(jax.jvp(lambda matrix: solve(matrix, events.append), (a,), (direction,)))
    for reference, value in zip(baseline, observed):
        np.testing.assert_array_equal(value, reference)
    assert events


def test_rpa_implicit_workload_includes_final_metric_action():
    a = _matrix()
    b = jnp.eye(a.shape[0], dtype=a.dtype) * 0.03
    events = []
    result = _ready(implicit_differential_davidson_lowest_tdhf(
        _rpa_operator(a, b), nroots=2, size=8, diag=jnp.diag(a),
        tol=1e-8, max_iter=20, max_subspace=8, workload_callback=events.append,
    ))
    assert bool(result[3])
    final_actions = [event for event in events
                     if event["event"] == "apply_end" and event["purpose"] == "implicit_metric"]
    assert len(final_actions) == 1
    final_action = final_actions[0]
    assert final_action["iteration"] is None
    assert final_action["subspace_dim"] is None
    assert final_action["columns"] == final_action["effective_columns"] == 2
    assert final_action["converged_roots"] == 2
    assert final_action["active_roots"] == 0
    assert events[-1] == final_action
    assert all(before["total_elapsed_seconds"] <= after["total_elapsed_seconds"]
               for before, after in zip(events, events[1:]))


def test_public_symmetric_workload_includes_final_original_operator_verification():
    from gradscf.solvers import solve_hermitian, EigenSolverConfig
    events=[]
    result=_ready(solve_hermitian(_matrix(),config=EigenSolverConfig(
        nroots=2,atol=1e-9,maxiter=20,max_subspace=8),workload_callback=events.append))
    assert bool(jnp.all(result.converged))
    final=[e for e in events if e['event']=='apply_end' and e['purpose']=='final_ritz_verify']
    assert len(final)==1 and final[0]['columns']==3
    assert max(final[0]['residual_norms'])<1e-9
