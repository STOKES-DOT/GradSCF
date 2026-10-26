"""Primitive-pool and free-contraction objectives use the same implicit RHF."""
import jax.numpy as jnp
import jax
import numpy as np

from gradscf.model.nnao import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys
from gradscf.model.nnao.training.pool import PoolRHF, minimize_reference


ROW = dict(structure_id='H2', symbols=['H', 'H'], charge=0, spin=0,
           coords_angstrom=[[0., 0., 0.], [0., 0., .74]], weight=1.)


def problem():
    layout = prepare_direct_basis(list(zip(ROW['symbols'], ROW['coords_angstrom'])),
                                 cart=False, basis_family='szp663_direct')
    keys = pool_channel_keys([layout])
    return PoolRHF(ROW, keys), jnp.zeros((len(keys), 2))


def test_pool_exponent_gradient_matches_reconverged_directional_difference():
    model, parameters = problem()
    parameters = parameters.at[0, 0].set(.02)
    energy, gradient, info = model.evaluate(parameters)
    direction = np.array([[.2, -.15], [.07, .11]])
    step = 1e-4
    plus = model.evaluate(parameters+step*direction, gradient=False)[0]
    minus = model.evaluate(parameters-step*direction, gradient=False)[0]
    np.testing.assert_allclose(np.sum(gradient*direction), (plus-minus)/(2*step),
                               atol=2e-6, rtol=2e-5)
    assert np.isfinite(energy) and info['converged'] and info['stable']
    assert info['pool_condition_number'] < 1e10


def test_free_contraction_stays_in_pool_and_lowers_reference_energy():
    model, parameters = problem()
    pool_energy, _, _ = model.evaluate(parameters, gradient=False)
    free = model.prepare_contraction(parameters)
    initial = free.layout.reference_outputs()
    result = minimize_reference(initial, free.evaluate, max_steps=3, learning_rate=.01)
    assert result['status'] in ('converged', 'max_steps')
    assert result['energy_hartree'] < result['initial_energy_hartree']
    assert result['energy_hartree'] >= pool_energy-1e-7
    assert result['info']['converged'] and result['info']['stable']


def test_fixed_contraction_device_gradient_matches_staged_implicit_pullback():
    model, parameters = problem()
    free = model.prepare_contraction(parameters)
    c = free.layout.reference_outputs()+.01*jnp.sin(jnp.arange(36).reshape(2, 3, 6))
    physical, pullback = jax.vjp(lambda x: free.project(x, *free.raw), c)
    (expected, _), cotangent = free.gradient(physical, free.enuc)
    expected_gradient = pullback(cotangent)[0]
    value, gradient, info = free.evaluate_device(c)
    assert isinstance(gradient, jax.Array)
    assert info['converged'] and info['stable']
    np.testing.assert_allclose(value, expected, atol=1e-11, rtol=0)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-9, rtol=1e-8)
    direction = np.cos(np.arange(c.size)).reshape(c.shape)
    step = 1e-4
    plus = free.evaluate(c+step*direction, gradient=False)[0]
    minus = free.evaluate(c-step*direction, gradient=False)[0]
    np.testing.assert_allclose(np.sum(gradient*direction), (plus-minus)/(2*step),
                               atol=2e-6, rtol=2e-5)


def test_failed_candidate_does_not_replace_accepted_reference_state():
    from gradscf.model.nnao.training.trainer import Adam
    def evaluate(x):
        if float(x[0]) < .5:
            raise RuntimeError('SCF failed')
        return float(x[0]**2), np.array([2*x[0]]), {'converged': True}
    result = minimize_reference(np.array([1.]), evaluate, max_steps=2,
                                learning_rate=10., max_backtracks=0)
    assert result['status'] == 'rejected'
    np.testing.assert_array_equal(result['parameters'], [1.])
    assert result['accepted_steps'] == 0 and result['history'][-1]['accepted'] is False
    optimizer = Adam(10.)
    assert result['optimizer'] == optimizer.encode(optimizer.init(np.array([1.])))


def test_reference_resume_preserves_adam_and_does_not_repeat_accepted_steps():
    evaluate = lambda x: (float(np.sum(x*x)), 2*x, {'converged': True})
    first = minimize_reference(np.array([1., -.4]), evaluate, max_steps=2)
    resumed = minimize_reference(first['parameters'], evaluate, max_steps=4, resume=first)
    uninterrupted = minimize_reference(np.array([1., -.4]), evaluate, max_steps=4)
    np.testing.assert_array_equal(resumed['parameters'], uninterrupted['parameters'])
    assert resumed['optimizer'] == uninterrupted['optimizer']
    assert [x['step'] for x in resumed['history']] == list(range(5))
    assert resumed['initial_energy_hartree'] == first['initial_energy_hartree']
