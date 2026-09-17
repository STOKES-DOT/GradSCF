from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import gto
from gradscf.pbc.scf import run_gamma_scf
from gradscf.integrals.periodic.fft import build_inputs
from gradscf.scf.autodiff import SCFDifferentiationConfig


@pytest.mark.parametrize('nk',[1,2])
@pytest.mark.parametrize('mode',['implicit','unrolled'])
def test_periodic_hf_bond_gradient_matches_finite_difference(mode,nk):
    cell=gto.M(atom='H 1 1 1; H 2.4 1 1',a=np.eye(3)*6,unit='Bohr',mesh=(25,)*3)
    from gradscf.pbc._kpoint import run_kpoint_scf
    kpts=cell.make_kpts((nk,1,1))
    config=SCFDifferentiationConfig(mode=mode,tolerance=1e-9,max_iter=40)
    def energy(delta):
        coords=cell.coords.at[1,0].add(delta)
        parameters=replace(cell.parameters,centers=coords,nuclear_coords=coords)
        inputs=build_inputs(cell,parameters=parameters,kpts=kpts)
        solver=run_gamma_scf if nk==1 else run_kpoint_scf
        result=solver(inputs,**({} if nk==1 else {'mesh':cell.mesh}),nelec=cell.nelec,differentiation=config,
                            conv_tol=1e-12,conv_tol_density=1e-10)
        return result.total_energy
    value,gradient=jax.jit(jax.value_and_grad(energy))(jnp.array(.03))
    step=1e-4
    fd=(energy(.03+step)-energy(.03-step))/(2*step)
    assert jnp.isfinite(value) and abs(gradient)>1e-3
    np.testing.assert_allclose(gradient,fd,atol=2e-6,rtol=0)
