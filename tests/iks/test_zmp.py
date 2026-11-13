"""Independent finite-lambda ZMP equations, spin factors and failure semantics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks


def problem(layout='rks'):
    s=np.array([[1.1,.05],[.05,.9]])
    values,vectors=np.linalg.eigh(s)
    x=(vectors/np.sqrt(values))@vectors.T
    def density(angle,occupation):
        c=x@np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        return occupation*np.outer(c[:,0],c[:,0]),s@(c*np.array([-.7,.8]))@c.T@s
    occupation=2 if layout=='rks' else 1
    actual,fock=density(.3,occupation);target,_=density(.1,occupation)
    if layout=='uks':
        actual_b,fock_b=density(-.2,1);target_b,_=density(.15,1)
        actual=np.stack((actual,actual_b));target=np.stack((target,target_b));fock=np.stack((fock,fock_b))
    factor=1 if layout=='rks' else 2
    base=fock-factor*.2*(actual-target)
    return iks.Target(target,s,2 if layout=='rks' else (1,1),layout=layout),base,actual


@pytest.mark.parametrize('layout',['rks','uks'])
def test_zmp_solves_the_finite_penalty_equation_and_returns_complete_orbitals(layout):
    target,base,expected=problem(layout)
    result=iks.invert(target,base_fock=base,method='zmp',coulomb=lambda d:d,
        lambdas=(0.,.1,.2),coulomb_tolerance=1.,scf_tolerance=1e-10,maxiter=250)
    assert result.converged and result.valid and result.reference.valid
    assert result.method=='zmp' and len(result.stages)==3
    np.testing.assert_allclose(result.density,expected,atol=2e-9)
    factor=1 if layout=='rks' else 2
    np.testing.assert_allclose(result.fock,base+factor*.2*(result.density-target.density),atol=2e-9)
    c=result.reference.coefficients;s=target.overlap
    np.testing.assert_allclose(c.swapaxes(-1,-2)@s@c,np.broadcast_to(np.eye(2),c.shape),atol=1e-10)
    np.testing.assert_allclose(result.fock@c,(s@c)*result.reference.energies[...,None,:],atol=1e-9)
    assert result.reference.xc_potential is None  # penalty is not an XC potential


def test_scf_convergence_does_not_certify_density_inversion():
    target,base,_=problem()
    result=iks.invert(target,base_fock=base,method='zmp',coulomb=lambda d:d,
        lambdas=(.2,),coulomb_tolerance=1e-14,scf_tolerance=1e-10)
    assert result.converged and not result.valid and result.reference is None
    assert result.coulomb_energy > 1e-14


@pytest.mark.parametrize('lambdas',[(1.,1.),(-1.,),(1.,float('nan')),()])
def test_zmp_requires_a_finite_increasing_penalty_schedule(lambdas):
    target,base,_=problem()
    with pytest.raises(ValueError,match='lambda|penalt'):
        iks.invert(target,base_fock=base,method='zmp',coulomb=lambda d:d,lambdas=lambdas)


def test_zmp_failure_and_inverse_map_ad_are_explicit():
    target,base,_=problem()
    result=iks.invert(target,base_fock=base,method='zmp',coulomb=lambda d:d,
        lambdas=(.2,),maxiter=1,scf_tolerance=1e-14)
    assert not result.converged and not result.valid and result.reference is None
    with pytest.raises(NotImplementedError,match='offline|inverse.map|AD'):
        jax.grad(lambda value:iks.invert(target,base_fock=base+value*jnp.eye(2),
            method='zmp',coulomb=lambda d:d).density.sum())(.1)


def test_xc_label_uses_recovered_hartree_and_grid_quality_is_independent():
    target,base,expected=problem()
    hcore=-.2*jnp.eye(2)
    grid=(jnp.eye(2),jnp.ones(2))
    result=iks.invert(target,base_fock=base,method='zmp',coulomb=lambda d:d,
        lambdas=(.2,),hcore=hcore,grid=grid,coulomb_tolerance=1.,density_tolerance=1e-14,
        scf_tolerance=1e-10)
    assert result.converged and not result.valid and result.reference is None
    np.testing.assert_allclose(result.xc_potential,result.fock-hcore-result.density,atol=1e-12)
    delta=np.diag(np.asarray(result.density-target.density))
    relative=np.linalg.norm(delta)/np.linalg.norm(np.diag(np.asarray(target.density)))
    np.testing.assert_allclose(result.relative_density_error,relative,atol=1e-12)


def test_empty_beta_population_and_invalid_coulomb_output():
    s=jnp.eye(2);density=jnp.stack((jnp.diag(jnp.array([1.,0.])),jnp.zeros((2,2))))
    target=iks.Target(density,s,(1,0),layout='uks')
    base=jnp.stack((jnp.diag(jnp.array([-.5,.5])),jnp.diag(jnp.array([-.4,.6]))))
    result=iks.invert(target,base_fock=base,method='zmp',coulomb=lambda d:d,lambdas=(1.,))
    assert result.valid and result.reference.valid
    np.testing.assert_array_equal(result.reference.occupations[1],0.)
    with pytest.raises(ValueError,match='Coulomb'):
        iks.invert(target,base_fock=base,method='zmp',coulomb=lambda d:jnp.array([1.,2.]))
