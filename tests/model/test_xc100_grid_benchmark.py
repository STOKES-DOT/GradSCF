"""Benchmark validation must reject nonfinite or inconsistent measurements."""
import jax.numpy as jnp
import numpy as np


def test_jit_timer_preserves_pytree_arguments_and_completed_results():
    from gradscf.model.autoxc.workflows.xc100.grid_benchmark import timed_jit
    result, timing = timed_jit(lambda p: {'value': p['x']**2},
                               ({'x': jnp.asarray([2., 3.])},), repeats=2)
    np.testing.assert_array_equal(result['value'], [4., 9.])
    assert result['value'].is_ready()
    assert len(timing['warm_seconds']) == 2
    assert timing['compile_seconds'] >= 0 and timing['trace_lower_seconds'] >= 0
    assert timing['first_execution_seconds'] >= 0


def test_array_comparison_never_hides_nonfinite_or_shape_errors():
    from gradscf.model.autoxc.workflows.xc100.grid_benchmark import compare_arrays
    assert compare_arrays([1., 2.], [1., 2.], atol=1e-8)['passed']
    assert not compare_arrays([1., 2.001], [1., 2.], atol=1e-8)['passed']
    assert not compare_arrays([np.nan], [np.nan], atol=1e-8)['passed']
    assert not compare_arrays([[1., 2.]], [1., 2.], atol=1e-8)['passed']
