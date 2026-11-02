"""Damped rescue reaches a stable raw-A Aufbau solution for the CaO pilot."""
import jax.numpy as jnp
import numpy as np

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys
from gradscf.model.nnao.training.pool import PoolRHF


def test_cao_stable_raw_scf_and_implicit_coefficient_gradient():
    row = dict(structure_id='CaO', symbols=['Ca','O'], charge=0, spin=0, weight=1,
               coords_angstrom=[[0.,0.,-.8987290136718362],[0.,0.,.8987290136718362]])
    layout = prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
        basis_family='nnao_szp_s',core_primitives='reference')
    keys=pool_channel_keys([layout])
    problem=PoolRHF(row,keys,basis_family='nnao_szp_s',core_primitives='reference')
    free=problem.prepare_contraction(jnp.zeros((len(keys),2)))
    c=layout.reference_outputs()
    energy,gradient,info=free.evaluate(c)
    assert info['converged'] and info['stable']
    assert info['fixed_point_residual'] < 1e-7 and info['orbital_residual'] < 1e-7
    assert energy < -750.98
    direction=np.cos(np.arange(c.size)).reshape(c.shape)
    direction/=np.linalg.norm(direction)
    step=1e-4
    plus=free.evaluate(c+step*direction)[0]
    minus=free.evaluate(c-step*direction)[0]
    np.testing.assert_allclose(np.sum(gradient*direction),(plus-minus)/(2*step),atol=2e-6,rtol=2e-5)
