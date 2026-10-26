"""One fixed-reference interface for analytic and learned total energies."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks
from test_thermal_inverse import problem


def entry():
    assert callable(getattr(iks, 'evaluate', None)), 'Missing inverse-reference functional evaluator'
    return iks.evaluate


@pytest.mark.parametrize('layout', ['rks', 'uks', 'gks'])
def test_energy_operator_gauge_and_parameter_derivatives(layout):
    evaluate = entry()
    target, basis, h = problem(layout)
    reference = iks.invert(target, basis, h, method='thermal_matrix', temperature=.2,
                           regularization=1e-12, tolerance=2e-7)
    s = target.overlap
    if layout == 'gks': s = jnp.kron(jnp.eye(2), s)
    number = sum(target.nelec) if layout == 'uks' else target.nelec
    def energy(p, d):
        return p*jnp.sum(jnp.abs(d)**2)+.03*jnp.einsum('...ij,...ji->', h, d).real
    def shifted(p, d):
        return energy(p, d)+.41*(jnp.einsum('ij,...ji->', s, d).real-number)
    actual = evaluate(reference, energy, .04)
    aligned = evaluate(reference, shifted, .04)
    assert actual.valid and aligned.valid
    np.testing.assert_allclose(actual.total_energy, aligned.total_energy, atol=2e-12)
    np.testing.assert_allclose(actual.operator_mse, aligned.operator_mse, atol=2e-12)
    np.testing.assert_allclose(actual.stationarity_mse, aligned.stationarity_mse, atol=2e-12)
    def loss(p):
        result = evaluate(reference, energy, p)
        return result.energy_error**2+.1*result.operator_mse
    derivative = jax.jit(jax.grad(loss))(.04)
    for step in (1e-3, 3e-4):
        finite = (loss(.04+step)-loss(.04-step))/(2*step)
        np.testing.assert_allclose(derivative, finite, atol=2e-9, rtol=2e-7)
    assert np.isfinite(derivative)


def test_nonfinite_model_is_not_turned_into_a_finite_score():
    evaluate = entry()
    target, basis, h = problem('rks')
    reference = iks.invert(target, basis, h, method='thermal_matrix', temperature=.2,
                           regularization=1e-12, tolerance=2e-7)
    result = evaluate(reference, lambda p, d: jnp.sum(d)*jnp.nan, None)
    assert not result.valid and not np.isfinite(result.total_energy)


def test_custom_analytic_energy_adapter_counts_coulomb_and_total_energy_once():
    from gradscf import dft
    assert callable(getattr(iks, 'energy_function', None)), 'Missing common analytic/neural energy adapter'
    density = jnp.stack((jnp.eye(2)*.6, jnp.eye(2)*.4))
    model = dft.custom_xc(input_fn=lambda fields: fields.density_matrix,
                         energy_fn=lambda p, d: p*jnp.sum(d*d))
    # Functional only needs these physical input fields for this formula.
    from gradscf.tools.features import MoleculeLikeState, MoleculeGridView
    state = MoleculeLikeState(ao=jnp.ones((1, 2)), ao_deriv1=jnp.zeros((4, 1, 2)),
        grid=MoleculeGridView(weights=jnp.ones(1)), rdm1=density,
        mo_coeff=jnp.stack((jnp.eye(2),)*2), mo_occ=jnp.ones((2, 2))*.5,
        mo_energy=jnp.zeros((2, 2)), h1e=-jnp.eye(2), rep_tensor=jnp.ones((2,)*4)*.1,
        overlap_matrix=jnp.eye(2))
    fn = iks.energy_function(state, model, layout='uks', nuclear_repulsion=0.)
    total = fn(.2, density)
    # one=-2, J=.5*.1*(sum D)^2=.2, XC=.2*(2*.36+2*.16)=.208
    np.testing.assert_allclose(total, -1.592, atol=2e-13)
    np.testing.assert_allclose(jax.grad(lambda p: fn(p, density))(.2), 1.04, atol=2e-13)


def test_energy_adapter_does_not_silently_omit_nuclear_repulsion():
    from types import SimpleNamespace
    from gradscf import dft
    state = SimpleNamespace(h1e=jnp.eye(2), rep_tensor=jnp.zeros((2,)*4))
    model = dft.custom_xc(input_fn=lambda fields: fields.density_matrix,
                          energy_fn=lambda p, d: jnp.sum(d))
    with pytest.raises(ValueError, match='nuclear'):
        iks.energy_function(state, model)


def test_plain_analytic_energy_protocol_needs_no_neural_specific_method():
    from types import SimpleNamespace
    density = jnp.stack((jnp.eye(2)*.6, jnp.eye(2)*.4))
    state = SimpleNamespace(h1e=-jnp.eye(2), rep_tensor=jnp.ones((2,)*4)*.1,
                            rdm1=density, nuclear_repulsion=.07)
    class Formula:
        def energy_from_molecule(self, params, current):
            return params*jnp.sum(current.rdm1**2)
    fn = iks.energy_function(state, Formula(), layout='uks')
    np.testing.assert_allclose(fn(.2, density), -1.522, atol=2e-13)


@pytest.mark.parametrize('fraction', [1., .3])
def test_named_recipe_exchange_is_included_once(fraction):
    from gradscf import dft
    from gradscf.tools.features import MoleculeLikeState, MoleculeGridView
    density = jnp.stack((jnp.eye(2)*.6, jnp.eye(2)*.4))
    state = MoleculeLikeState(ao=jnp.ones((1, 2)), ao_deriv1=jnp.zeros((4, 1, 2)),
        grid=MoleculeGridView(weights=jnp.ones(1)), rdm1=density,
        mo_coeff=jnp.stack((jnp.eye(2),)*2), mo_occ=jnp.ones((2, 2))*.5,
        mo_energy=jnp.zeros((2, 2)), h1e=-jnp.eye(2), rep_tensor=jnp.ones((2,)*4)*.1,
        overlap_matrix=jnp.eye(2))
    model = dft.custom_xc(f'{fraction}*HF')
    fn = iks.energy_function(state, model, nuclear_repulsion=.07)
    np.testing.assert_allclose(fn(None, density), -1.8-.104*fraction+.07, atol=2e-13)
