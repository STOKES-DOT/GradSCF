"""Additional topology and policy coverage, separate from the initial RED set."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from test_grid_locality import _provider, _require_locality


def test_changed_static_primitive_and_angular_topology_rebuilds_bounds_and_values():
    locality = _require_locality()
    raw = _provider()

    def evaluate(grid):
        bounds = locality.ao_block_bounds(grid, grid.coords, grid.weights, deriv=2)
        indices, valid, overflow = locality.select_local_aos(bounds, cutoff=0., capacity=6)
        values = locality.evaluate_local_ao(grid, grid.coords, grid.weights,
                                            indices, valid, deriv=2)
        return bounds, values[0], values[1], values[2]

    compiled = jax.jit(evaluate)
    original = compiled(raw)
    variants = [replace(raw, nprims=(1, *raw.nprims[1:])),
                replace(raw, angulars=((3, 0, 0), *raw.angulars[1:]))]
    for variant in variants:
        actual = compiled(variant)
        reference = variant.block(0, variant.ngrids, deriv=2)
        assert not np.allclose(actual[0][0], original[0][0])
        assert not np.allclose(actual[1][:, 0], original[1][:, 0])
        for value, expected in zip(actual[1:], reference[:3]):
            np.testing.assert_allclose(value, expected, atol=2e-13, rtol=2e-13)


def test_nonfinite_bounds_force_active_instead_of_silent_screening():
    locality = _require_locality()
    indices, valid, overflow = locality.select_local_aos(
        jnp.asarray([jnp.nan, jnp.inf, 0., 1e-30]), cutoff=1e-12, capacity=3)
    np.testing.assert_array_equal(np.asarray(indices)[np.asarray(valid)], [0, 1])
    assert not bool(overflow)
