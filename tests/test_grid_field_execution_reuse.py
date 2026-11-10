"""Small physical grid fields reuse compilation with live differentiable inputs.

Five fixed quadrature points, two Cartesian Gaussians, float64 and a three-point
block isolate execution reuse. These tests do not benchmark molecular SCF.
"""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids.blocks import AOGrid
from gradscf.tools.features import (
    grid_features_with_spin_gradients_for_molecule,
    restricted_grid_features_with_gradients,
    restricted_grid_response_variables,
    unrestricted_grid_features_with_gradients,
)


_MODES = ("restricted", "unrestricted", "spin", "LDA", "GGA", "MGGA", "MGGA_LAPL")


@contextmanager
def _backend_compilations(monkeypatch):
    from jax._src import compiler

    original = compiler.backend_compile_and_load
    events = []

    def count(*args, **kwargs):
        events.append(1)
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(compiler, "backend_compile_and_load", count)
        yield events


def _reference(parameters, *, streamed, restricted=False):
    coords = jnp.asarray([[.1, -.2, .3], [-.3, .4, .2], [.5, -.1, .2],
                          [-.2, -.3, .6], [.4, .3, -.2]])
    coords = coords + parameters[0] * jnp.asarray([.1, -.2, .3])
    centers = jnp.asarray([[.1, -.1, .2], [.6, .2, -.3]])
    centers = centers + parameters[1] * jnp.asarray([-.2, .1, .3])
    provider = AOGrid(
        coords=coords,
        weights=jnp.asarray([.2, .4, .1, .3, .15]) * jnp.exp(parameters[5]),
        centers=centers,
        exponents=jnp.asarray([[.8], [1.1]]) * jnp.exp(parameters[2]),
        coefficients=jnp.asarray([[1.], [.7]]) * (1. + parameters[3]),
        nprims=(1, 1), angulars=((0, 0, 0), (1, 0, 0)), block_size=3,
    )
    density = jnp.asarray([[.7, .08], [.08, .4]]) * (1. + parameters[4])
    density = 2. * density if restricted else jnp.stack((density, .63 * density))
    if streamed:
        ao, derivative, laplacian = jnp.zeros((0, 2)), jnp.zeros((4, 0, 2)), None
    else:
        ao, derivative, laplacian, _, _ = provider.block(0, 5, deriv=2)
    return SimpleNamespace(
        ao=ao, ao_deriv1=derivative, ao_laplacian=laplacian,
        rdm1=density, grid_ao=provider if streamed else None,
        grid=SimpleNamespace(coords=coords, weights=provider.weights),
    )


def _fields(molecule, mode):
    if mode == "restricted":
        return restricted_grid_features_with_gradients(molecule)
    if mode == "unrestricted":
        return unrestricted_grid_features_with_gradients(molecule)
    if mode == "spin":
        return grid_features_with_spin_gradients_for_molecule(molecule)
    return restricted_grid_response_variables(molecule, feature_kind=mode)


def _point():
    return jnp.asarray([.04, -.03, .08, -.02, .06, .02, .3])


@pytest.mark.parametrize("mode", _MODES)
def test_grid_fields_reuse_compilation_with_changed_provider_and_density(monkeypatch, mode):
    first = _reference(_point(), streamed=True, restricted=mode == "restricted")
    changed = _reference(_point() + .07, streamed=True, restricted=mode == "restricted")
    original = jax.block_until_ready(_fields(first, mode))
    with _backend_compilations(monkeypatch) as events:
        repeated = jax.block_until_ready(_fields(first, mode))
        updated = jax.block_until_ready(_fields(changed, mode))
    assert not events, f"Same-shape physical grid fields triggered {len(events)} compilations"
    for before, repeat in zip(jax.tree.leaves(original), jax.tree.leaves(repeated), strict=True):
        np.testing.assert_allclose(before, repeat, atol=2e-12, rtol=2e-12)
    assert any(not np.allclose(before, after) for before, after in
               zip(jax.tree.leaves(original), jax.tree.leaves(updated), strict=True))
    dense = _reference(_point() + .07, streamed=False, restricted=mode == "restricted")
    for actual, expected in zip(jax.tree.leaves(updated), jax.tree.leaves(_fields(dense, mode)), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("mode", ("restricted", "spin", "MGGA_LAPL"))
def test_grid_fields_preserve_basis_density_weight_and_neural_chain_rule(mode):
    def loss(parameters, *, streamed):
        molecule = _reference(parameters, streamed=streamed, restricted=mode == "restricted")
        # A nonlinear downstream model consumes only physical fields. Geometry,
        # exponents, contractions, D, weights and the model scale all stay live.
        return sum(jnp.dot(molecule.grid.weights,
                           jnp.sum(jnp.tanh(parameters[6] * field).reshape(5, -1)**2, axis=1))
                   for field in jax.tree.leaves(_fields(molecule, mode)))

    streamed = lambda p: loss(p, streamed=True)
    dense = lambda p: loss(p, streamed=False)
    point = _point()
    direction = jnp.asarray([.1, -.2, .1, .15, -.1, .2, .1])
    actual, actual_jvp = jax.jvp(streamed, (point,), (direction,))
    expected, expected_jvp = jax.jvp(dense, (point,), (direction,))
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(actual_jvp, expected_jvp, atol=3e-11, rtol=3e-11)
    gradient = jax.grad(streamed)(point)
    np.testing.assert_allclose(gradient, jax.grad(dense)(point), atol=3e-11, rtol=3e-11)
    assert np.all(np.abs(np.asarray(gradient)) > 1e-8)


def test_grid_fields_keep_tail_and_block_structure_dynamic(monkeypatch):
    reference = _reference(_point(), streamed=True)
    original = jax.block_until_ready(_fields(reference, "MGGA_LAPL"))
    reference.grid_ao = replace(reference.grid_ao, block_size=2)
    with _backend_compilations(monkeypatch) as events:
        changed = jax.block_until_ready(_fields(reference, "MGGA_LAPL"))
    assert events, "Changing block shape selects a different grid program"
    for actual, expected in zip(jax.tree.leaves(changed), jax.tree.leaves(original), strict=True):
        assert actual.shape[0] == 5
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    with _backend_compilations(monkeypatch) as events:
        jax.block_until_ready(_fields(reference, "MGGA_LAPL"))
    assert not events


@pytest.mark.parametrize("mode", ("restricted", "spin", "MGGA_LAPL"))
def test_opaque_map_provider_retains_grid_feature_contract(mode):
    reference = _reference(_point(), streamed=True, restricted=mode == "restricted")
    expected = _fields(reference, mode)

    class OpaqueGrid:
        """An external map provider need not expose a JAX PyTree contract."""

        def __init__(self, raw):
            self.raw = raw

        def map(self, function, *, deriv=0):
            return self.raw.map(function, deriv=deriv)

    reference.grid_ao = OpaqueGrid(reference.grid_ao)
    for actual, value in zip(jax.tree.leaves(_fields(reference, mode)),
                             jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual, value, atol=2e-12, rtol=2e-12)
