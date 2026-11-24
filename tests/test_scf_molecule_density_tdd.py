"""Density contracts for molecular references and Neural XC predictors."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.neural_xc.training import (
    density_on_grid,
    density_on_grid_spin_resolved,
)
from gradscf.scf.molecules import UnrestrictedMolecule
from test_streamed_response_grid import _molecular_reference, _reference


def _molecule(position, *, streamed, unrestricted):
    view = _reference(position, streamed=streamed, unrestricted=unrestricted)
    return _molecular_reference(view, unrestricted=unrestricted)


@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
@pytest.mark.parametrize("streamed", [False, True], ids=["dense", "streamed"])
@pytest.mark.parametrize("spin_resolved", [False, True], ids=["total", "spin-resolved"])
def test_molecular_grid_density_shapes_and_values(unrestricted, streamed, spin_resolved):
    dense = _molecule(.04, streamed=False, unrestricted=unrestricted)
    molecule = _molecule(.04, streamed=streamed, unrestricted=unrestricted)
    expected = np.einsum("spq,rp,rq->rs", dense.rdm1, dense.ao, dense.ao)
    ngrids = dense.grid.weights.size
    if streamed:
        assert molecule.ao.shape[0] == 0
        assert ngrids % molecule.grid_ao.block_size != 0
    public_density = molecule.density()
    assert public_density.shape == ((ngrids,) if unrestricted else (ngrids, 2))
    np.testing.assert_allclose(
        public_density, expected.sum(axis=1) if unrestricted else expected,
        rtol=2e-12, atol=2e-12,
    )

    read_density = density_on_grid_spin_resolved if spin_resolved else density_on_grid
    actual = read_density(molecule)
    assert actual.shape == ((ngrids, 2) if spin_resolved else (ngrids,))
    np.testing.assert_allclose(
        actual, expected if spin_resolved else expected.sum(axis=1),
        rtol=2e-12, atol=2e-12,
    )


def test_unrestricted_spin_density_skips_discarded_total_contraction(monkeypatch):
    molecule = _molecule(.04, streamed=False, unrestricted=True)
    evaluate_total = UnrestrictedMolecule.density
    total_calls = []

    def counted_total(self):
        total_calls.append(self)
        return evaluate_total(self)

    monkeypatch.setattr(UnrestrictedMolecule, "density", counted_total)
    actual = density_on_grid_spin_resolved(molecule)
    expected = np.einsum("spq,rp,rq->rs", molecule.rdm1, molecule.ao, molecule.ao)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert not total_calls


@pytest.mark.parametrize("unrestricted", [False, True], ids=["restricted", "unrestricted"])
def test_streamed_training_density_preserves_geometry_and_density_ad(unrestricted):
    def evaluate(position, scale, *, streamed):
        molecule = _molecule(position, streamed=streamed, unrestricted=unrestricted)
        matrix = molecule.rdm1.at[0].multiply(scale)
        return density_on_grid_spin_resolved(replace(molecule, rdm1=matrix))

    def dense(position, scale):
        return evaluate(position, scale, streamed=False)

    def streamed(position, scale):
        return evaluate(position, scale, streamed=True)

    inputs = (jnp.asarray(.04), jnp.asarray(1.1))
    tangents = (jnp.asarray(.13), jnp.asarray(-.2))
    value, tangent = jax.jvp(jax.jit(streamed), inputs, tangents)
    expected_value, expected_tangent = jax.jvp(dense, inputs, tangents)
    assert value.shape == tangent.shape == (7, 2)
    np.testing.assert_allclose(value, expected_value, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-11, atol=2e-11)

    def loss(function, position, scale):
        return jnp.sum(function(position, scale) ** 2)

    actual_grad = jax.grad(loss, argnums=(1, 2))(streamed, *inputs)
    expected_grad = jax.grad(loss, argnums=(1, 2))(dense, *inputs)
    assert all(abs(component) > 1e-8 for component in expected_grad)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("spin_resolved", [False, True], ids=["total-hook", "spin-hook"])
def test_custom_density_hook_remains_authoritative(spin_resolved):
    calls = []
    hook_value = jnp.asarray([[.3, .7], [.2, .4], [.5, .8]])
    if not spin_resolved:
        hook_value = hook_value[:, 0]

    def density():
        calls.append(True)
        return hook_value

    molecule = SimpleNamespace(density=density)
    if spin_resolved:
        molecule.rdm1 = jnp.zeros((2, 1, 1))
        molecule.ao = jnp.zeros((3, 1))
    expected = hook_value if spin_resolved else hook_value[:, None]
    np.testing.assert_array_equal(density_on_grid_spin_resolved(molecule), expected)
    assert len(calls) == 1
    np.testing.assert_array_equal(density_on_grid(molecule), expected.sum(axis=1))
    assert len(calls) == 2


@pytest.mark.parametrize("has_total_hook", [False, True], ids=["raw-arrays", "legacy-total-hook"])
def test_custom_rank_three_density_retains_dense_spin_fallback(has_total_hook):
    native = _molecule(.04, streamed=False, unrestricted=True)
    custom = SimpleNamespace(ao=native.ao, rdm1=native.rdm1)
    if has_total_hook:
        custom.density = lambda: native.density()
    expected = np.einsum("spq,rp,rq->rs", native.rdm1, native.ao, native.ao)
    np.testing.assert_allclose(density_on_grid_spin_resolved(custom), expected, rtol=2e-12, atol=2e-12)


def test_custom_unrestricted_subclass_keeps_its_spin_density_hook():
    expected = jnp.arange(14, dtype=jnp.float64).reshape(7, 2)

    class CustomMolecule(UnrestrictedMolecule):
        def density(self):
            return expected

    molecule = CustomMolecule(**vars(_molecule(.04, streamed=False, unrestricted=True)))
    np.testing.assert_array_equal(density_on_grid_spin_resolved(molecule), expected)
