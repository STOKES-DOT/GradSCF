"""The anthracene reverse JIT must take saved VJP residuals as dynamic inputs."""
from pathlib import Path
import runpy
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np


def _load_cartesian_pullback(monkeypatch, tmp_path):
    folder = Path(__file__).resolve().parents[1]/'examples/tdscf/anthracene_ht'
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(folder))
        patch.setenv('GRADSCF_ANTHRACENE_INPUT', str(folder/'input_eight_states.json'))
        patch.setenv('GRADSCF_ANTHRACENE_OUTPUT', str(tmp_path/'raw'))
        patch.setenv('GRADSCF_ANTHRACENE_FD_OUTPUT', str(tmp_path/'fd'))
        patch.delitem(sys.modules, 'common', raising=False)
        try:
            example = runpy.run_path(str(folder/'05_ad_response.py'),
                                    run_name='anthracene_pullback_test')
        finally:
            # Remove the temporary import; the context restores any prior
            # common module together with sys.path and the environment.
            sys.modules.pop('common', None)
    return example['cartesian_pullback']


def test_saved_pullback_residuals_are_dynamic_in_the_actual_example(
        monkeypatch, tmp_path, record_property):
    reverse = _load_cartesian_pullback(monkeypatch, tmp_path)

    @jax.jit
    def values(coordinates, coefficients):
        field = jnp.sin(coordinates)*coefficients
        return jnp.stack((jnp.sum(field), jnp.sum(field**2)))

    point = jnp.linspace(-.4, .6, 1025, dtype=jnp.float64)
    coefficients = jnp.linspace(.8, 1.3, point.size, dtype=jnp.float64)
    weights = (jnp.array([.3, -.7]), jnp.array([-1.1, .2]))
    pullbacks = [jax.vjp(values, point+shift, coefficients+.3*shift)[1]
                 for shift in (0., .2)]

    # Positive control: the failed production wiring places the saved arrays
    # inside the compiled closure rather than among its runtime arguments.
    first_pullback = pullbacks[0]
    closed_reverse = jax.jit(lambda weight: first_pullback(weight)[0])
    option = 'jax_captured_constants_warn_bytes'
    previous = getattr(jax.config, option)
    try:
        jax.config.update(option, 4096)
        with warnings.catch_warnings(record=True) as old_warnings:
            warnings.simplefilter('always')
            closed_reverse.lower(weights[0])
        with warnings.catch_warnings(record=True) as dynamic_warnings:
            warnings.simplefilter('always')
            lowered = reverse.lower(first_pullback, weights[0])
        captured = lambda messages: [str(item.message) for item in messages
                                    if 'constants were captured' in str(item.message)]
        old_captures, dynamic_captures = captured(old_warnings), captured(dynamic_warnings)
        assert old_captures, 'The old closure must reproduce the captured-residual lowering warning'
        assert not dynamic_captures, '\n'.join(dynamic_captures)
    finally:
        jax.config.update(option, previous)
    # Inspect the actual lowered input signature rather than assuming where a
    # nested JIT stores captured constants in a ClosedJaxpr representation.
    dynamic_residuals = [aval for aval in jax.tree.leaves(lowered.in_avals)
                         if getattr(aval, 'shape', None) == point.shape]
    assert dynamic_residuals, 'Saved vector residuals must enter the reverse JIT as array arguments'
    record_property('captured_constants_warn_bytes', 4096)
    record_property('old_closure_capture_warnings', '\n'.join(old_captures))
    record_property('dynamic_capture_warning_count', len(dynamic_captures))
    record_property('dynamic_vector_residual_inputs', len(dynamic_residuals))
    record_property('pullback_tree_structure_equal',
        jax.tree.structure(pullbacks[0]) == jax.tree.structure(pullbacks[1]))

    results = []
    for pullback_index, pullback in enumerate(pullbacks):
        warm_entries = None
        for weight_index, weight in enumerate(weights):
            expected = pullback(weight)[0]
            actual = reverse(pullback, weight)
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
            if weight_index == 0:
                results.append(np.asarray(actual))
            if warm_entries is None:
                warm_entries = reverse._cache_size()
                record_property(f'pullback_{pullback_index}_warm_cache_entries', warm_entries)
            assert reverse._cache_size() == warm_entries, (
                f'Pullback {pullback_index}, cotangent {weight_index} unexpectedly recompiled '
                'while reusing the same saved tape')
    assert np.max(abs(results[1]-results[0])) > 1e-4, 'A new numerical origin must update the pullback'
