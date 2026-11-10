"""Potential metrics retain spin channels, gauge and failure semantics."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks


def test_potential_metric_is_explicit_and_spin_resolved():
    metric = getattr(iks, 'potential_error', None)
    assert metric is not None, 'Missing shared potential metric'
    predicted = jnp.array([[1., -1.], [-1., 1.]])
    target = jnp.zeros((2, 2))
    weights = jnp.ones(2)
    rho = jnp.ones((2, 2))
    # Both spin averages and both spatial means vanish, but grid error does not.
    np.testing.assert_allclose(metric(predicted, target, weights, metric='grid_l2'), 2.)
    np.testing.assert_allclose(metric(predicted, target, weights, metric='density_integral', density=rho), 0.)
    shifted = target+jnp.array([[.2], [-.3]])
    np.testing.assert_allclose(metric(shifted, target, weights, metric='grid_l2', gauge='weighted_mean'), 0., atol=1e-15)
    with pytest.raises(ValueError, match='gauge'):
        metric(predicted, target, weights, metric='density_integral', density=rho, gauge='weighted_mean')


def test_nonfinite_potential_never_becomes_a_zero_label():
    metric = getattr(iks, 'potential_error', None)
    assert metric is not None, 'Missing shared potential metric'
    assert not np.isfinite(metric(jnp.ones(2), jnp.array([jnp.nan, 0.]), jnp.ones(2)))


def test_restricted_density_integral_accepts_one_grid_channel():
    value = iks.potential_error(jnp.array([1., 2.]), jnp.zeros(2), jnp.array([1., 2.]),
        metric='density_integral', density=jnp.array([.5, 1.]))
    np.testing.assert_allclose(value, 4.5**2)
