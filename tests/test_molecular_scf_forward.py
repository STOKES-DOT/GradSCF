"""Existing DIIS forward plus the same fully live implicit stationary response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('xc',['hf','pbe'])
def test_scf_forward_matches_minimum_state_and_complete_geometry_response(xc):
    from gradscf import gto,dft
    from gradscf_tools.molecular_response import MolecularEnergy
    mf=dft.RKS(gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='sto-3g',cart=True),
        xc=xc,grids_level=1,conv_tol=1e-12,conv_tol_grad=1e-10).run()
    old=MolecularEnergy(mf,gradient_tolerance=1e-9)
    new=MolecularEnergy(mf,gradient_tolerance=1e-9,forward='scf')
    for r in (new.coordinates,new.coordinates.at[1,2].add(.01)):
        previous=old.solve(r);updated=new.solve(r)
        assert bool(previous.stationary) and bool(updated.stationary)
        np.testing.assert_allclose(updated.total_energy,previous.total_energy,atol=2e-10,rtol=0)
        np.testing.assert_allclose(updated.density_matrix,previous.density_matrix,atol=2e-7,rtol=0)
    r=new.coordinates
    gradient=jax.jit(jax.grad(new))(r)
    previous=jax.jit(jax.grad(old))(r)
    np.testing.assert_allclose(gradient,previous,atol=1e-6,rtol=1e-6)
    direction=jnp.zeros_like(r).at[1,2].set(1.)
    value=jax.jit(new)
    for step in (2e-4,1e-4):
        fd=(value(r+step*direction)-value(r-step*direction))/(2*step)
        np.testing.assert_allclose(jnp.vdot(gradient,direction),fd,atol=1e-5,rtol=1e-5)


def test_scf_seed_strategy_does_not_redefine_explicit_iteration_derivatives():
    from gradscf import gto,scf
    from gradscf_tools.molecular_response import MolecularEnergy
    mf=scf.RHF(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g',cart=True)).run()
    with pytest.raises(ValueError,match='implicit'):
        MolecularEnergy(mf,mode='explicit',forward='scf')


def test_scf_seed_strategy_rejects_unrestricted_reference():
    from gradscf import gto,scf
    from gradscf_tools.molecular_response import MolecularEnergy
    mf=scf.UHF(gto.M(atom='H 0 0 0',basis='sto-3g',spin=1,cart=True)).run()
    with pytest.raises(ValueError,match='restricted'):
        MolecularEnergy(mf,forward='scf')
