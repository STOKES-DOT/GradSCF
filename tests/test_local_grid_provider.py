"""Local block reduction, exact dense fallback, and differentiable support.

Coordinates are in Bohr, exponents in Bohr^-2, and all checks use float64.
Positive cutoffs constrain AO channels, not the consumer's energy error.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids import blocks


def _local_class():
    assert hasattr(blocks, "_LocalAOGrid"), (
        "A private local AO provider must reduce compact blocks with exact overflow fallback"
    )
    return blocks._LocalAOGrid


def _raw(parameters=None):
    p = jnp.zeros(6) if parameters is None else parameters
    centers = jnp.asarray([[0., 0., 0.], [.2, -.1, .3], [-.3, .2, -.1],
                           [16., 0., 0.], [16.2, -.1, .3], [15.7, .2, -.1]])
    centers = centers + p[0] * jnp.asarray([.2, -.1, .3])
    # Positive padded exponents isolate the provider from dense-kernel padding AD.
    exponents = jnp.asarray([[.7, 1.4], [.9, 1.], [1.1, 2.],
                             [.7, 1.4], [.9, 1.], [1.1, 2.]])
    coefficients = jnp.asarray([[.6, -.2], [.8, 0.], [.5, .3],
                                [.6, -.2], [.8, 0.], [.5, .3]])
    coords = jnp.asarray([[16.1, .3, -.2], [.1, .3, -.2], [15.6, .2, .1],
                          [-.4, .2, .1], [.3, -.2, .5], [16.3, -.2, .5],
                          [.2, .1, -.3]])
    coords = coords + p[3] * jnp.asarray([.1, .2, -.2])
    weights = jnp.asarray([.2, .1, -.04, .3, .15, .12, .21])*jnp.exp(p[4])
    return blocks.AOGrid(coords, weights, centers, exponents*jnp.exp(p[1]),
                         coefficients*(1+p[2]), (2, 1, 2, 2, 1, 2),
                         ((0, 0, 0), (1, 0, 0), (0, 2, 0))*2, 4)


def _density(raw, scale=0.):
    vector = jnp.arange(raw.nao, dtype=raw.coords.dtype)/17+.2
    return (jnp.eye(raw.nao)*.6 + jnp.outer(vector, vector)) * jnp.exp(scale)


def _moments(grid, density, *, deriv):
    """Quadratic density energy and its full-AO density derivative."""
    initial = (jnp.asarray(0., dtype=density.dtype), jnp.zeros_like(density))

    def consume(carry, ao, derivative, laplacian, coords, weights, indices):
        features = ao
        if deriv:
            features = features + .02*jnp.sum(derivative[1:], axis=0)
        if deriv == 2:
            features = features + .03*laplacian
        local_dm = density[indices[:, None], indices[None, :]]
        rho = jnp.einsum("gi,ij,gj->g", features, local_dm, features)
        local_v = jnp.einsum("g,gi,gj->ij", weights*rho, features, features)
        return (carry[0]+.5*jnp.dot(weights, rho**2),
                carry[1].at[indices[:, None], indices[None, :]].add(local_v))

    if isinstance(grid, blocks.AOGrid):
        return grid.reduce(lambda carry, *args: consume(
            carry, *args, jnp.arange(grid.nao)), initial, deriv=deriv)
    return grid.reduce_local(consume, initial, deriv=deriv)


@pytest.mark.parametrize("deriv", [0, 1, 2])
@pytest.mark.parametrize("cutoff,capacity", [(0., 1), (1e-12, 3), (1e-12, 1)])
def test_local_energy_and_full_matrix_scatter_match_dense_with_tail(deriv, cutoff, capacity):
    cls = _local_class()
    raw = _raw()
    local = cls.from_grid(raw, cutoff=cutoff, capacity=capacity)
    density = _density(raw)
    actual = jax.jit(lambda grid: _moments(grid, density, deriv=deriv))(local)
    expected = _moments(raw, density, deriv=deriv)
    for value, reference in zip(actual, expected):
        assert np.isfinite(np.asarray(value)).all()
        np.testing.assert_allclose(value, reference, atol=2e-12, rtol=2e-12)
    # The scattered matrix really is the derivative with respect to full D.
    gradient = jax.grad(lambda d: _moments(local, d, deriv=deriv)[0])(density)
    np.testing.assert_allclose(actual[1], gradient, atol=2e-12, rtol=2e-12)


def test_external_grid_map_and_generic_reduction_preserve_original_order():
    cls = _local_class()
    raw = _raw()
    local = cls.from_grid(raw, cutoff=1e-12, capacity=3)
    np.testing.assert_array_equal(local.coords, raw.coords)
    np.testing.assert_array_equal(local.weights, raw.weights)
    np.testing.assert_array_equal(jnp.sort(local.order), jnp.arange(raw.ngrids))
    assert not np.array_equal(local.order, jnp.arange(raw.ngrids))
    assert local.nao == raw.nao and local.ngrids == raw.ngrids
    assert local.block_size == raw.block_size and local.adaptive == raw.adaptive
    feature = lambda ao, derivative, laplacian, coords, weights: (coords, weights, ao.sum(-1))
    actual, expected = local.map(feature), raw.map(feature)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)
    for value, reference in zip(local.block(1, 3, deriv=2), raw.block(1, 3, deriv=2)):
        np.testing.assert_array_equal(value, reference)
    consume = lambda carry, ao, derivative, laplacian, coords, weights: carry+ao.shape[-1]
    assert int(local.reduce(consume, jnp.asarray(0))) == 2*raw.nao


@pytest.mark.parametrize("cutoff,capacity,expected_width", [
    (1e-12, 3, 3), (1e-12, 1, 6), (0., 1, 6), (1e10, 3, 3),
])
def test_executed_consumer_has_compact_width_or_exact_full_fallback(
        cutoff, capacity, expected_width):
    cls = _local_class()
    raw = _raw()
    local = cls.from_grid(raw, cutoff=cutoff, capacity=capacity)

    def consume(carry, ao, derivative, laplacian, coords, weights, indices):
        assert ao.shape[-1] == indices.shape[0]
        assert derivative.shape[-1] == laplacian.shape[-1] == ao.shape[-1]
        return carry + jnp.sum(weights)*ao.shape[-1]

    actual = jax.jit(lambda g: g.reduce_local(consume, jnp.asarray(0.), deriv=2))(local)
    np.testing.assert_allclose(actual, raw.weights.sum()*expected_width, atol=1e-13)


def test_padded_unused_slots_scatter_zero_even_with_duplicate_indices():
    cls = _local_class()
    raw = _raw()
    local = cls.from_grid(raw, cutoff=1e-12, capacity=5)
    actual = _moments(local, _density(raw), deriv=2)
    expected = _moments(raw, _density(raw), deriv=2)
    for value, reference in zip(actual, expected):
        assert np.isfinite(np.asarray(value)).all()
        np.testing.assert_allclose(value, reference, atol=2e-12, rtol=2e-12)


def test_pytree_numeric_updates_recompute_support_with_retained_bijective_order():
    cls = _local_class()
    raw = _raw()
    local = cls.from_grid(raw, cutoff=1e-12, capacity=3)
    # The former far AO becomes near; the first tile must now overflow.
    moved = replace(raw, centers=raw.centers.at[3].set(jnp.asarray([.1, 0., 0.])))
    moved = replace(moved, coords=moved.coords[::-1], weights=moved.weights[::-1])
    updated = replace(local, raw=moved)
    leaves, structure = jax.tree_util.tree_flatten(updated)
    updated = jax.tree_util.tree_unflatten(structure, leaves)
    np.testing.assert_array_equal(updated.order, local.order)
    compiled = jax.jit(lambda grid: _moments(grid, _density(raw), deriv=2))
    for value, reference in zip(compiled(updated), _moments(moved, _density(raw), deriv=2)):
        assert np.isfinite(np.asarray(value)).all()
        np.testing.assert_allclose(value, reference, atol=2e-12, rtol=2e-12)


def test_density_geometry_basis_grid_weight_ad_matches_dense_and_two_step_fd():
    cls = _local_class()
    p = jnp.asarray([.03, -.02, .01, .04, -.03, .02])
    direction = jnp.asarray([.2, -.3, .1, .4, -.2, .3])

    def observable(parameters, *, local):
        raw = _raw(parameters)
        grid = cls.from_grid(raw, cutoff=1e-12, capacity=3) if local else raw
        return _moments(grid, _density(raw, parameters[5]), deriv=2)[0]

    results = []
    for use_local in (True, False):
        function = lambda x: observable(x, local=use_local)
        results.append(jax.jit(lambda x: (
            function(x), jax.grad(function)(x),
            jax.jvp(function, (x,), (direction,))[1],
            jax.jvp(jax.grad(function), (x,), (direction,))[1]))(p))
    for value, reference in zip(*results):
        assert np.isfinite(np.asarray(value)).all()
        assert np.isfinite(np.asarray(reference)).all()
        np.testing.assert_allclose(value, reference, atol=2e-11, rtol=2e-10)
    local = jax.jit(lambda x: observable(x, local=True))
    gradient = jax.jit(jax.grad(local))
    for h in (1e-4, 5e-5):
        fd = (local(p+h*direction)-local(p-h*direction))/(2*h)
        hfd = (gradient(p+h*direction)-gradient(p-h*direction))/(2*h)
        assert np.isfinite(np.asarray(hfd)).all()
        np.testing.assert_allclose(fd, results[0][2], atol=2e-9, rtol=2e-7)
        np.testing.assert_allclose(hfd, results[0][3], atol=2e-8, rtol=2e-6)


def test_empty_grid_and_zero_weights_are_safe():
    cls = _local_class()
    raw = _raw()
    for empty in (replace(raw, coords=raw.coords[:0], weights=raw.weights[:0]),
                  replace(raw, weights=jnp.zeros_like(raw.weights))):
        local = cls.from_grid(empty, cutoff=1e-12, capacity=3)
        energy, matrix = _moments(local, _density(raw), deriv=2)
        assert np.isfinite(np.asarray(matrix)).all()
        assert float(energy) == 0.
        np.testing.assert_array_equal(matrix, np.zeros((raw.nao, raw.nao)))


def test_nonfinite_ao_is_retained_and_cannot_be_hidden_by_pruning():
    cls = _local_class()
    raw = _raw()
    raw = replace(raw, coefficients=raw.coefficients.at[5, 0].set(jnp.nan))
    local = cls.from_grid(raw, cutoff=1e-12, capacity=3)
    energy, matrix = _moments(local, _density(raw), deriv=2)
    assert not np.isfinite(np.asarray(energy)).all()
    assert not np.isfinite(np.asarray(matrix)).all()


@pytest.mark.parametrize("policy", [
    {"cutoff": -1., "capacity": 3}, {"cutoff": float("nan"), "capacity": 3},
    {"cutoff": float("inf"), "capacity": 3}, {"cutoff": 0., "capacity": -1},
    {"cutoff": 0., "capacity": 1.5}, {"cutoff": 0., "capacity": 3, "cell_size": 0.},
])
def test_invalid_local_policy_is_rejected(policy):
    cls = _local_class()
    with pytest.raises(ValueError):
        cls.from_grid(_raw(), **policy)
