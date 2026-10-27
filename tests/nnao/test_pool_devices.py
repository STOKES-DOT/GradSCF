"""Native integral derivatives stay on CPU; SCF can run on an explicit device."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys
from gradscf.model.nnao.training.pool import PoolRHF


ROW=dict(structure_id='H2',symbols=['H','H'],coords_angstrom=[[0.,0.,0.],[0.,0.,.74]],
         charge=0,spin=0,weight=1.)


def setup(device):
    with jax.default_device(jax.devices('cpu')[0]):
        layout=prepare_direct_basis(list(zip(ROW['symbols'],ROW['coords_angstrom'])),
            basis_family='nnao_dzp_s',core_primitives='6-31g')
    keys=pool_channel_keys([layout])
    model=PoolRHF(ROW,keys,basis_family='nnao_dzp_s',core_primitives='6-31g',scf_device=device)
    return model,jnp.full((len(keys),2),.01)


def test_explicit_cpu_pool_device_and_free_contraction():
    model,parameters=setup(jax.devices('cpu')[0])
    energy,gradient,info=model.evaluate(parameters)
    assert np.isfinite(energy) and np.isfinite(gradient).all() and info['stable']
    assert info['scf_backend']=='cpu'
    assert model.enuc.device.platform=='cpu'
    free=model.prepare_contraction(parameters)
    value,derivative,details=free.evaluate(free.layout.reference_outputs())
    assert details['stable'] and np.isfinite(derivative).all()
    assert value >= energy-1e-7


def test_gpu_pool_matches_cpu_exponent_and_contraction_gradients():
    try:
        gpu=jax.devices('gpu')[0]
    except RuntimeError:
        pytest.skip('CUDA allocation required')
    cpu=jax.devices('cpu')[0]
    baseline,parameters=setup(cpu)
    hybrid,_=setup(gpu)
    reference=baseline.evaluate(parameters)
    result=hybrid.evaluate(parameters)
    assert result[2]['scf_backend']=='gpu'
    np.testing.assert_allclose(result[0],reference[0],atol=1e-10,rtol=0)
    np.testing.assert_allclose(result[1],reference[1],atol=1e-8,rtol=1e-7)
    direction=np.cos(np.arange(parameters.size)).reshape(parameters.shape)
    direction/=np.linalg.norm(direction);step=1e-4
    plus=hybrid.evaluate(parameters+step*direction,gradient=False)[0]
    minus=hybrid.evaluate(parameters-step*direction,gradient=False)[0]
    np.testing.assert_allclose(np.sum(result[1]*direction),(plus-minus)/(2*step),atol=2e-6,rtol=2e-5)
    a,b=baseline.prepare_contraction(parameters),hybrid.prepare_contraction(parameters)
    assert b.raw[0].device.platform=='gpu' and hybrid.enuc.device.platform=='cpu'
    coefficients=a.layout.reference_outputs()
    left,right=a.evaluate(coefficients),b.evaluate(coefficients)
    np.testing.assert_allclose(left[0],right[0],atol=1e-10,rtol=0)
    np.testing.assert_allclose(left[1],right[1],atol=1e-8,rtol=1e-7)
