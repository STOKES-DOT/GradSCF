"""Bound-XC finalization keeps the historical level-shift orbital-energy protocol."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from gradscf.scf import uks
from gradscf.solvers.scf import SCFState


def test_bound_xc_level_shift_corrects_eps_before_required_final_eval(monkeypatch):
    calls, selected = [], {}
    base_a, base_b = jnp.diag(jnp.array([-1.3, .7])), jnp.diag(jnp.array([-1.1, .8]))
    def values(eps_a, eps_b):
        total = jnp.sum(eps_a)+jnp.sum(eps_b)
        return 10.+total, 2.*total, base_a+.01*total*jnp.eye(2), base_b-.02*total*jnp.eye(2)
    def raw(**kwargs):
        eps_a, eps_b = kwargs['mo_energy_a'], kwargs['mo_energy_b']
        calls.append((eps_a, eps_b))
        return values(eps_a, eps_b)
    def solve(**kwargs):
        dm, coeff = kwargs['density_spin'], kwargs['mo_coeff_spin']
        shifted_eps = jnp.array([[2., 3.], [4., 5.]])
        energy, fock, aux = kwargs['state_evaluator'](dm, coeff, shifted_eps, None)
        state = SCFState(dm, coeff, shifted_eps, energy, fock, aux)
        selected['state'] = state
        legacy = (dm, coeff, shifted_eps, fock, jnp.asarray(True), jnp.asarray(2),
                  jnp.zeros(2), jnp.asarray(2), jnp.asarray(2), jnp.asarray(0.), jnp.asarray(0.))
        return (*legacy, state) if kwargs.get('return_state', False) else legacy
    monkeypatch.setattr(uks, '_raw_fock_and_energy_for_state', raw)
    monkeypatch.setattr(uks, 'run_unrestricted_scf', solve)
    result = uks.run_uks_from_integrals(overlap=jnp.eye(2), hcore=jnp.diag(jnp.array([-1., .5])),
        eri=jnp.zeros((2, 2, 2, 2)), nalpha=1, nbeta=1, nuclear_repulsion=0.,
        ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
        bound_xc=SimpleNamespace(exact_exchange_fraction=0.),
        config=uks.UKSConfig(xc_spec='hf', max_cycle=2, level_shift=.2))
    state = selected['state']
    corrected = jnp.stack([jnp.diag(c.T @ f @ c) for c, f in zip(state.mo_coeff, state.fock)])
    expected_e, expected_xc, expected_fa, expected_fb = values(*corrected)
    # A bound callback may independently use eps in its energy AND potential.
    np.testing.assert_allclose(jnp.stack(calls[-1]), corrected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.mo_energy_alpha, corrected[0], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.mo_energy_beta, corrected[1], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.total_energy, expected_e, atol=1e-13, rtol=0)
    np.testing.assert_allclose(result.xc_energy, expected_xc, atol=1e-13, rtol=0)
    np.testing.assert_allclose(result.fock_matrix_alpha, expected_fa, atol=1e-13, rtol=0)
    np.testing.assert_allclose(result.fock_matrix_beta, expected_fb, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(result.density_matrix_alpha, state.density[0])
    np.testing.assert_array_equal(result.density_matrix_beta, state.density[1])
