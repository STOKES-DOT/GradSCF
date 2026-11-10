"""Inactive primitive padding must have finite, zero parameter derivatives.

These regressions supplement the frozen locality RED set. An independent
unpadded oracle removes dummy primitives rather than accepting matched NaNs.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids.ao import _evaluate_cartesian_ao_kernel
from test_grid_locality import _provider, _observable


_POINT = jnp.asarray([.03, -.02, .01, .04, -.03])
_DIRECTION = jnp.asarray([.2, -.3, .1, .4, -.2])


def _responses(function, point):
    return (function(point), jax.grad(function)(point),
            jax.jvp(function, (point,), (_DIRECTION,))[1],
            jax.jvp(jax.grad(function), (point,), (_DIRECTION,))[1])


def _assert_finite(values):
    for label, value in zip(("value", "VJP", "JVP", "HVP"), values):
        assert np.isfinite(np.asarray(value)).all(), f"non-finite AO padding {label}: {value}"


@pytest.mark.parametrize("local", [False, True])
def test_dense_and_local_parameter_responses_are_all_finite(local):
    function = lambda p: _observable(p, local=local)
    actual = jax.jit(lambda p: _responses(function, p))(_POINT)
    _assert_finite(actual)
    for step in (1e-4, 5e-5):
        high, low = _POINT + step*_DIRECTION, _POINT - step*_DIRECTION
        fd = (function(high)-function(low))/(2*step)
        hfd = (jax.grad(function)(high)-jax.grad(function)(low))/(2*step)
        np.testing.assert_allclose(fd, actual[2], atol=2e-8, rtol=2e-7, equal_nan=False)
        np.testing.assert_allclose(hfd, actual[3], atol=2e-7, rtol=2e-6, equal_nan=False)


def _kernel_energy(p, *, deriv, padded):
    raw = _provider(p)
    angulars = jnp.asarray(raw.angulars, dtype=jnp.int32)
    if padded:
        values = _evaluate_cartesian_ao_kernel(
            raw.coords, raw.centers, raw.exponents, raw.coefficients,
            jnp.asarray(raw.nprims), angulars, deriv=deriv)
    else:
        columns = []
        for index, count in enumerate(raw.nprims):
            columns.append(_evaluate_cartesian_ao_kernel(
                raw.coords, raw.centers[index:index+1],
                raw.exponents[index:index+1, :count],
                raw.coefficients[index:index+1, :count],
                jnp.asarray([count]), angulars[index:index+1], deriv=deriv))
        values = jnp.concatenate(columns, axis=-1)
    features = jnp.sum(values**2, axis=-1)
    if deriv:
        features = jnp.sum(features, axis=0)
    return jnp.dot(raw.weights, features)


@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_padded_parameter_ad_matches_unpadded_primitive_oracle(deriv):
    padded = lambda p: _kernel_energy(p, deriv=deriv, padded=True)
    unpadded = lambda p: _kernel_energy(p, deriv=deriv, padded=False)
    actual = jax.jit(lambda p: _responses(padded, p))(_POINT)
    reference = jax.jit(lambda p: _responses(unpadded, p))(_POINT)
    _assert_finite(reference)
    _assert_finite(actual)
    for value, expected in zip(actual, reference):
        np.testing.assert_allclose(value, expected, atol=2e-11, rtol=2e-11,
                                   equal_nan=False)
