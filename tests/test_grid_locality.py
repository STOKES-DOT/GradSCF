"""Opt-in AO locality: exact permutation, dynamic support, reduced AO work.

Bohr coordinates, Bohr^-2 exponents, float64. The positive cutoff bounds AO
channels only; it is neither an energy nor a potential error guarantee.
"""
from dataclasses import replace
import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids.blocks import AOGrid


def _require_locality():
    module = "gradscf.integrals.grids.locality"
    assert importlib.util.find_spec(module) is not None, (
        "AO locality module must supply reversible ordering and reduced AO evaluation"
    )
    return importlib.import_module(module)


def _provider(parameters=None):
    p = jnp.zeros(5) if parameters is None else parameters
    centers = jnp.asarray([[0., 0., 0.], [.2, -.1, .3], [-.3, .2, -.1],
                           [12., 1., 0.], [-13., 0., 1.], [0., 14., 0.]])
    centers = centers + p[0] * jnp.asarray([.2, -.1, .3])
    exponents = jnp.asarray([[.7, 1.4], [.9, 0.], [1.1, 2.],
                             [1.2, 0.], [.8, 1.5], [1.1, 0.]])
    coefficients = jnp.asarray([[.6, -.2], [.8, 0.], [.5, .3],
                                [.7, 0.], [.6, .2], [.9, 0.]])
    coords = jnp.asarray([[.1, .3, -.2], [-.4, .2, .1], [.3, -.2, .5],
                          [0., 0., 0.], [.2, .1, -.3]])
    coords = coords + p[3] * jnp.asarray([.1, .2, -.2])
    weights = jnp.asarray([.2, .1, -.04, .3, .15]) * jnp.exp(p[4])
    return AOGrid(coords, weights, centers, exponents*jnp.exp(p[1]),
                  coefficients*(1+p[2]), (2, 1, 2, 1, 2, 1),
                  ((0, 0, 0), (1, 0, 0), (0, 2, 0),
                   (0, 0, 0), (1, 1, 1), (0, 0, 1)), 3)


def test_spatial_permutation_is_reversible_stable_and_keeps_voxels_together():
    locality = _require_locality()
    coords = jnp.asarray([[4.1, 0., 0.], [.2, .1, 0.], [-3.7, .2, 0.],
                          [4.1, 0., 0.], [.3, .2, .1], [-3.8, .1, 0.]])
    weights = jnp.asarray([.1, -.2, .3, .4, 0., .6])
    order, inverse = jax.jit(locality.spatial_grid_permutation)(coords)
    np.testing.assert_array_equal(jnp.sort(order), jnp.arange(6))
    np.testing.assert_array_equal(coords[order][inverse], coords)
    np.testing.assert_array_equal(weights[order][inverse], weights)
    cells = np.floor(np.asarray(coords[order])/4).astype(int)
    assert np.array_equal(cells[0], cells[1])
    assert np.array_equal(cells[2], cells[3])
    assert np.array_equal(cells[4], cells[5])
    assert int(inverse[0]) < int(inverse[3])  # duplicate points retain input order
    direction = jnp.arange(18, dtype=coords.dtype).reshape(6, 3)/19

    def roundtrip(x):
        indices, restore = locality.spatial_grid_permutation(x)
        return x[indices][restore]

    tangent = jax.jvp(roundtrip, (coords,), (direction,))[1]
    np.testing.assert_array_equal(tangent, direction)
    empty = locality.spatial_grid_permutation(jnp.zeros((0, 3)))
    assert empty[0].shape == empty[1].shape == (0,)


@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_dynamic_box_bounds_cover_values_derivatives_and_laplacian(deriv):
    locality = _require_locality()
    raw = _provider()
    # The envelope must handle s/p/d/f, mixed-sign contractions and padded
    # zero exponents. Include the derivative maxima away from the AO center.
    coords = jnp.asarray([[x, y, z] for x in (-.7, 0., .8)
                          for y in (-.6, .1, .9) for z in (-.5, 0., .6)])
    raw = replace(raw, coords=coords, weights=jnp.ones(len(coords)))
    bounds = jax.jit(lambda grid: locality.ao_block_bounds(
        grid, grid.coords, grid.weights, deriv=deriv))(raw)
    block = raw.block(0, len(coords), deriv=deriv)
    channels = block[0][None] if deriv == 0 else block[1]
    if deriv == 2:
        channels = jnp.concatenate((channels, block[2][None]), axis=0)
    measured = jnp.max(jnp.abs(channels), axis=(0, 1))
    assert np.isfinite(np.asarray(bounds)).all()
    assert np.all(np.asarray(bounds) + 1e-14 >= np.asarray(measured))
    assert np.all(np.asarray(bounds[3:]) < 1e-12)


def test_bounds_recompute_numeric_support_and_ignore_only_explicit_padding():
    locality = _require_locality()
    raw = _provider()
    bound = jax.jit(lambda grid, point_mask=None: locality.ao_block_bounds(
        grid, grid.coords, grid.weights, deriv=2, point_mask=point_mask))
    original = bound(raw)
    moved = replace(raw, centers=raw.centers.at[3].set(jnp.zeros(3)))
    diffuse = replace(raw, exponents=raw.exponents.at[3, 0].set(.002))
    rescaled = replace(raw, coefficients=raw.coefficients.at[0].multiply(4))
    assert float(bound(moved)[3]) > .1
    assert float(bound(diffuse)[3]) > float(original[3])*1e10
    np.testing.assert_allclose(bound(rescaled)[0], 4*original[0], rtol=2e-13)
    padded = replace(raw, coords=jnp.concatenate((raw.coords, raw.centers[3:4])),
                     weights=jnp.concatenate((raw.weights, jnp.zeros(1))))
    point_mask = jnp.arange(padded.ngrids) < raw.ngrids
    np.testing.assert_array_equal(bound(padded, point_mask), original)
    # Without an explicit mask, the added zero-weight point is a real point.
    assert float(bound(padded)[3]) > .1
    negative = replace(padded, weights=padded.weights.at[-1].set(-.1))
    assert float(bound(negative)[3]) > .1
    np.testing.assert_array_equal(
        bound(replace(raw, weights=jnp.zeros_like(raw.weights))), original)
    np.testing.assert_array_equal(
        bound(raw, jnp.zeros(raw.ngrids, dtype=bool)), np.zeros(raw.nao))


@pytest.mark.parametrize("point_mask", [[True], [1, 1, 1, 1, 1]])
def test_block_bounds_reject_invalid_point_mask(point_mask):
    locality = _require_locality()
    raw = _provider()
    with pytest.raises(ValueError, match="point_mask"):
        locality.ao_block_bounds(raw, raw.coords, raw.weights, point_mask=point_mask)


@pytest.mark.parametrize("cutoff,capacity,expected,overflow", [
    (1e-4, 4, [0, 2], False), (1e-4, 1, [0], True),
    (0., 4, [0, 1, 2, 3], False), (2., 2, [], False),
])
def test_local_selection_handles_padding_overflow_and_zero_cutoff(
        cutoff, capacity, expected, overflow):
    locality = _require_locality()
    bounds = jnp.asarray([.2, 0., .3, 1e-8])
    indices, valid, actual_overflow = jax.jit(lambda x: locality.select_local_aos(
        x, cutoff=cutoff, capacity=capacity))(bounds)
    assert indices.shape == valid.shape == (capacity,)
    np.testing.assert_array_equal(np.asarray(indices)[np.asarray(valid)], expected)
    assert bool(actual_overflow) is overflow


@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_local_ao_evaluates_only_selected_columns(monkeypatch, deriv):
    locality = _require_locality()
    raw = _provider()
    indices, valid, overflow = locality.select_local_aos(
        locality.ao_block_bounds(raw, raw.coords, raw.weights, deriv=deriv),
        cutoff=1e-10, capacity=4)
    assert not bool(overflow)
    expected = raw.block(0, raw.ngrids, deriv=deriv)
    original = locality._evaluate_cartesian_ao_kernel
    seen = []

    def checked(coords, centers, exponents, coefficients, nprims, angulars, **kwargs):
        seen.append(centers.shape[0])
        assert centers.shape[0] == 4 < raw.nao
        return original(coords, centers, exponents, coefficients, nprims, angulars, **kwargs)

    monkeypatch.setattr(locality, "_evaluate_cartesian_ao_kernel", checked)
    block = locality.evaluate_local_ao(raw, raw.coords, raw.weights, indices, valid,
                                       deriv=deriv)
    assert seen == [4]
    for actual, dense in zip(block[:3], expected[:3]):
        if dense is not None:
            reference = jnp.take(dense, indices, axis=-1)*valid
            np.testing.assert_allclose(actual, reference, atol=2e-13, rtol=2e-13)
            assert actual.shape[-1] == 4
    np.testing.assert_array_equal(block[3], raw.coords)
    np.testing.assert_array_equal(block[4], raw.weights)


def _observable(p, *, local):
    raw = _provider(p)
    if local:
        locality = _require_locality()
        indices, valid, overflow = locality.select_local_aos(
            locality.ao_block_bounds(raw, raw.coords, raw.weights, deriv=2),
            cutoff=1e-10, capacity=3)
        ao, derivative, laplacian, _, weights = locality.evaluate_local_ao(
            raw, raw.coords, raw.weights, indices, valid, deriv=2)
    else:
        ao, derivative, laplacian, _, weights = raw.block(0, raw.ngrids, deriv=2)
        ao, derivative, laplacian = ao[:, :3], derivative[:, :, :3], laplacian[:, :3]
    return jnp.dot(weights, jnp.sum(.2*ao**2+.03*laplacian**2
                                    +.04*jnp.sum(derivative[1:]**2, axis=0), axis=-1))


def test_local_jvp_vjp_hvp_and_two_step_finite_differences():
    p = jnp.asarray([.03, -.02, .01, .04, -.03])
    direction = jnp.asarray([.2, -.3, .1, .4, -.2])
    local = lambda x: _observable(x, local=True)
    dense = lambda x: _observable(x, local=False)
    results = []
    for function in (local, dense):
        results.append(jax.jit(lambda x: (
            function(x), jax.grad(function)(x),
            jax.jvp(function, (x,), (direction,))[1],
            jax.jvp(jax.grad(function), (x,), (direction,))[1]))(p))
    for actual, reference in zip(results[0], results[1]):
        np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=2e-11)
    for h in (1e-4, 5e-5):
        fd = (local(p+h*direction)-local(p-h*direction))/(2*h)
        np.testing.assert_allclose(fd, results[0][2], atol=2e-8, rtol=2e-7)
        hfd = (jax.grad(local)(p+h*direction)-jax.grad(local)(p-h*direction))/(2*h)
        np.testing.assert_allclose(hfd, results[0][3], atol=2e-7, rtol=2e-6)


def test_empty_ao_provider_and_empty_grid_have_explicit_shapes():
    locality = _require_locality()
    raw = _provider()
    empty_grid = replace(raw, coords=jnp.zeros((0, 3)), weights=jnp.zeros((0,)))
    np.testing.assert_array_equal(locality.ao_block_bounds(
        empty_grid, empty_grid.coords, empty_grid.weights, deriv=2), np.zeros(6))
    empty_ao = replace(raw, centers=raw.centers[:0], exponents=raw.exponents[:0],
                       coefficients=raw.coefficients[:0], nprims=(), angulars=())
    indices, valid, overflow = locality.select_local_aos(
        jnp.zeros((0,)), cutoff=1e-10, capacity=0)
    assert not bool(overflow)
    block = locality.evaluate_local_ao(empty_ao, raw.coords, raw.weights,
                                       indices, valid, deriv=2)
    assert block[0].shape == block[2].shape == (5, 0)
    assert block[1].shape == (4, 5, 0)


@pytest.mark.parametrize("kwargs", [{"cutoff": -1., "capacity": 2},
                                      {"cutoff": 1e-8, "capacity": -1}])
def test_local_selection_rejects_invalid_policy(kwargs):
    locality = _require_locality()
    with pytest.raises(ValueError):
        locality.select_local_aos(jnp.ones(4), **kwargs)
