"""Explicit SCF seeds change the search branch, not the implicit derivative."""
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np


def test_seed_is_dynamic_and_has_no_physical_response(monkeypatch):
    objective = import_module('gradscf.model.nnao.training.objective')
    objective._df_rhf_solver.cache_clear()
    traces = []
    original = objective._run_scf_with_rescue

    def traced(solver, kwargs, config, **options):
        traces.append(kwargs['init_density'].shape)
        return original(solver, kwargs, config, **options)

    monkeypatch.setattr(objective, '_run_scf_with_rescue', traced)
    value, value_grad = objective._df_rhf_solver(2, 1e-8, 0.)
    physical = (jnp.array([[1., .08], [.08, 1.1]]),
                jnp.array([[-1., -.12], [-.12, .3]]),
                jnp.array([[[.15, .02], [.02, .11]], [[.04, -.01], [-.01, .03]]]))
    seed = jnp.eye(2)
    first, gradient = value_grad(physical, .2, init_density=seed)
    second, other = value_grad(physical, .2, init_density=.8 * seed)
    assert traces == [(2, 2)]
    assert bool(first[1]['converged']) and bool(second[1]['converged'])
    np.testing.assert_allclose(first[0], second[0], atol=1e-11, rtol=0)
    for actual, expected in zip(gradient, other):
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
    response = jax.grad(lambda d: value(physical, .2, init_density=d)[0])(seed)
    np.testing.assert_array_equal(response, np.zeros((2, 2)))


def test_fixed_pool_seed_preserves_energy_and_coefficient_gradient(tmp_path):
    from gradscf.model.nnao.basis import prepare_direct_basis
    from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
    from gradscf.model.nnao.training.initial_guess import atomic_guess_data, atomic_density_guess
    row = dict(name='H2', structure_id='H2/seeded', symbols=['H', 'H'],
               coords_angstrom=[[0., 0., 0.], [0., 0., .74]], charge=0, spin=0)
    layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                                  basis_family='nnao_szp_s', core_primitives='6-31g')
    experiment, _ = prepare_fixed_pool(row, layout, tmp_path / 'pool.npz', storage='host')
    coefficients = layout.reference_outputs()
    overlap = experiment.project(coefficients, *experiment.raw)[0]
    seed = atomic_density_guess(overlap, *atomic_guess_data(layout))
    reference, reference_gradient, _ = experiment.evaluate(coefficients)
    actual, gradient, details = experiment.evaluate(coefficients, init_density=seed)
    assert details['converged'] and details['stable']
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=0)
    np.testing.assert_allclose(gradient, reference_gradient, atol=1e-8, rtol=1e-8)
