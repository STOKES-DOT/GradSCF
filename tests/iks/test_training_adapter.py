"""The ordinary Trainer consumes inverse references without a new optimizer."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft
from gradscf.dft import iks
from gradscf.model.neural_xc import training
from gradscf.tools.features import MoleculeLikeState, MoleculeGridView
from test_orbital_reference import reference


def state_and_model():
    ref = reference('uks')
    state = MoleculeLikeState(ao=jnp.array([[1., .2], [.3, .8]]),
        ao_deriv1=jnp.zeros((4, 2, 2)), grid=MoleculeGridView(weights=jnp.ones(2)),
        rdm1=ref.target.density, mo_coeff=ref.coefficients,
        mo_occ=ref.occupations, mo_energy=ref.energies, h1e=jnp.zeros((2, 2)),
        rep_tensor=jnp.zeros((2,)*4), overlap_matrix=ref.target.overlap)
    # An external energy callback with two independent continuous parameters.
    model = dft.custom_xc(input_fn=lambda f: f.density_matrix,
        energy_fn=lambda p, d: p[0]*jnp.sum(d*d)+p[1]*jnp.sum(d[..., 0, 1]))
    return ref, state, model


def test_existing_trainer_uses_the_same_inverse_objective_and_history():
    ref, state, model = state_and_model()
    sample = training.Sample(state, inverse=ref, nuclear_repulsion=0.)
    np.testing.assert_allclose(sample.molecule.rdm1, ref.density, atol=2e-13)
    assert sample.inverse is ref
    params = jnp.array([.1, .2])
    weights = {'energy': {'mse': 1., 'mae': 1.}, 'stationarity': {'norm2': .2}}
    fn = iks.energy_function(sample.molecule, model, layout='uks', nuclear_repulsion=0.)
    expected, _ = iks.loss(ref, fn, params, weights=weights)
    trainer = training.Trainer(model, params=params)
    trainer.mode = 'fixed_density'
    trainer.loss = weights
    trainer.learning_rate = .01
    initial = trainer.evaluate(sample)
    np.testing.assert_allclose(initial['loss'], expected, atol=2e-12)
    trainer.run(sample, steps=4)
    assert len(trainer.history['loss']) == 5
    assert trainer.history['loss'][-1] < float(initial['loss'])
    assert trainer.metrics['scf_converged'] is None
    assert trainer.metrics['response_converged'] is None
    assert np.isnan(trainer.metrics['response_mse'])
    assert int(trainer.metrics['scf_cycles']) == 0
    assert trainer.metrics['inverse_valid']
    assert 'stationarity_norm2' in trainer.metrics
    assert all(trainer.history['update_accepted'][1:])


def test_inverse_adapter_does_not_silently_run_scf_for_unsupported_losses():
    ref, state, model = state_and_model()
    sample = training.Sample(state, inverse=ref, nuclear_repulsion=0.)
    trainer = training.Trainer(model, params=jnp.array([.1, .2]))
    trainer.mode = 'implicit'
    with pytest.raises(ValueError, match='inverse|fixed'):
        trainer.evaluate(sample)
    trainer.mode = 'fixed_density'
    trainer.loss = {'density': {'mse': 1.}}
    with pytest.raises(ValueError, match='density|self.consistent'):
        trainer.evaluate(sample)


def test_inverse_adapter_rejects_a_different_ao_metric():
    from dataclasses import replace
    ref, state, _ = state_and_model()
    with pytest.raises(ValueError, match='overlap|basis|metric'):
        training.Sample(replace(state, overlap_matrix=jnp.eye(2)), inverse=ref, nuclear_repulsion=0.)


def test_nonfinite_candidate_rejects_the_existing_adam_update():
    ref, state, _ = state_and_model()
    model = dft.custom_xc(input_fn=lambda f: f.density_matrix,
        energy_fn=lambda p, d: p['strength']*jnp.sum(d)*jnp.nan)
    sample = training.Sample(state, inverse=ref, nuclear_repulsion=0.)
    trainer = training.Trainer(model, params={'strength': jnp.asarray(.1)})
    trainer.run(sample, steps=1)
    np.testing.assert_allclose(trainer.params['strength'], .1)
    assert trainer.history['update_accepted'][-1] is False
    assert trainer.history['optimizer_step'][-1] == 0


def test_external_neural_energy_retains_mixed_parameter_density_ad():
    ref, state, _ = state_and_model()
    model = dft.custom_xc(input_fn=lambda f: f.density_matrix.reshape(-1),
        energy_fn=lambda p, x: jnp.sum(jnp.tanh(x@p['w']+p['b'])))
    params = {'w': jnp.arange(16, dtype=float).reshape(8, 2)*.01,
              'b': jnp.array([.1, -.2])}
    sample = training.Sample(state, inverse=ref, nuclear_repulsion=0.)
    fn = iks.energy_function(sample.molecule, model, layout='uks', nuclear_repulsion=0.)
    def loss(p): return iks.loss(ref, fn, p, weights={'stationarity': {'norm2': 1.}})[0]
    direction = jax.tree.map(jnp.ones_like, params)
    actual = jax.jit(lambda p: jax.jvp(loss, (p,), (direction,))[1])(params)
    def displaced(h): return jax.tree.map(lambda p, v: p+h*v, params, direction)
    finite = (loss(displaced(1e-5))-loss(displaced(-1e-5)))/2e-5
    np.testing.assert_allclose(actual, finite, atol=2e-8, rtol=2e-6)


def test_common_energy_adapter_exposes_a_live_xc_component():
    ref, state, model = state_and_model()
    fn = iks.energy_function(state, model, layout='uks', nuclear_repulsion=0., component='xc')
    p = jnp.array([.1, .2])
    expected = model.energy_for_density(p, state, ref.density)
    np.testing.assert_allclose(fn(p, ref.density), expected, atol=2e-13)


@pytest.mark.parametrize('fraction', [1., .3])
def test_named_xc_component_does_not_include_hartree_or_nuclear_energy(fraction):
    from dataclasses import replace
    ref, state, _ = state_and_model()
    state = replace(state, rep_tensor=jnp.ones((2,)*4)*.1)
    fn = iks.energy_function(state, dft.custom_xc(f'{fraction}*HF'), layout='uks',
        nuclear_repulsion=.07, component='xc')
    expected = -.05*fraction*jnp.sum(jnp.sum(ref.density, axis=(-2, -1))**2)
    np.testing.assert_allclose(fn(None, ref.density), expected, atol=2e-13)


@pytest.mark.parametrize('label,objective', [('xc_potential', 'xc_potential'), ('density', 'predicted_density')])
def test_inverse_does_not_silently_ignore_an_active_sample_label(label, objective):
    from dataclasses import replace
    ref, state, model = state_and_model()
    ref = replace(ref, xc_potential=jnp.zeros_like(ref.density))
    explicit = jnp.ones_like(ref.density) if label == 'xc_potential' else jnp.ones(2)
    sample = training.Sample(state, inverse=ref, nuclear_repulsion=0., **{label: explicit})
    trainer = training.Trainer(model, params=jnp.array([.1, .2]))
    assert np.isfinite(trainer.evaluate(sample)['loss'])  # inactive labels are immaterial
    trainer.loss = {objective: {'mse' if objective == 'xc_potential' else 'norm2': 1.}}
    with pytest.raises(ValueError, match='label|target|reference'):
        trainer.evaluate(sample)
