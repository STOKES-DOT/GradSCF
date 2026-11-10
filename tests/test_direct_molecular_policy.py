"""Relaxed molecular direct AD retains requested screening and XC boundaries."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, scf
from gradscf.integrals.plan import IntegralPlan
from gradscf_tools.molecular_response import MolecularEnergy
from test_hfks_acceptance import _molecule


CASES=[('RHF','hf'),('UHF','hf'),('RKS','svwn'),('RKS','pbe0'),
       ('RKS','cam-b3lyp'),('UKS','pbe')]


def mean_field(method, xc, backend, *, grid_screen_tol=0.):
    options=dict(jk_backend=backend,direct_scf_tol=1e-12,direct_scf_workers=2,
        init_guess='hcore',grids_level=0,max_cycle=150,conv_tol=1e-12,
        conv_tol_density=1e-10,conv_tol_grad=1e-9,max_memory=1024.,
        grid_screen_tol=grid_screen_tol)
    if method.endswith('KS'):
        options['xc']=xc
    mf=getattr(dft if method.endswith('KS') else scf,method)(_molecule(method),**options).run()
    assert mf.converged
    return mf


@pytest.mark.parametrize('method,xc',CASES)
def test_relaxed_direct_gradient_preserves_cutoff_and_matches_full(method,xc,monkeypatch):
    direct_mf=mean_field(method,xc,'direct')
    direct=MolecularEnergy(direct_mf,forward='scf',jk_backend='direct',gradient_tolerance=1e-7)
    # The numerical seed and the implicit stationary energy must receive the
    # user's nonzero policy rather than silently disabling all screening.
    assert direct.scf_config.direct_scf_tol==direct_mf.direct_scf_tol
    reference=MolecularEnergy(mean_field(method,xc,'full'),forward='scf',gradient_tolerance=1e-7)
    r=direct.coordinates
    direction=.05*jnp.cos(jnp.arange(r.size).reshape(r.shape)+.2)
    reference_value,reference_gradient=jax.jit(jax.value_and_grad(reference))(r)
    reference_scalar=jax.jit(reference)
    steps=(1e-4,3e-5)
    expected_fd=[(reference_scalar(r+h*direction)-reference_scalar(r-h*direction))/(2*h) for h in steps]
    original=IntegralPlan.evaluate
    def forbid_eri(self,operator,*args,**kwargs):
        if operator=='eri':
            raise AssertionError('Relaxed direct geometry must not construct full/packed AO ERIs')
        return original(self,operator,*args,**kwargs)
    monkeypatch.setattr(IntegralPlan,'evaluate',forbid_eri)
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    get_jk=NativeDirectBasis.get_jk
    seen=[]
    def observe(self,density,**kwargs):
        seen.append(kwargs.get('screening_threshold',0.))
        return get_jk(self,density,**kwargs)
    monkeypatch.setattr(NativeDirectBasis,'get_jk',observe)
    value,gradient=jax.jit(jax.value_and_grad(direct))(r)
    assert np.isfinite(np.asarray(gradient)).all() and np.isfinite(float(value))
    np.testing.assert_allclose(value,reference_value,atol=1e-8,rtol=0.)
    np.testing.assert_allclose(gradient,reference_gradient,atol=1e-6,rtol=1e-5)
    derivative=jnp.vdot(gradient,direction)
    scalar=jax.jit(direct)
    for h,expected in zip(steps,expected_fd):
        fd=(scalar(r+h*direction)-scalar(r-h*direction))/(2*h)
        np.testing.assert_allclose(derivative,fd,atol=1e-6,rtol=1e-4)
        np.testing.assert_allclose(derivative,expected,atol=1e-6,rtol=1e-4)
    assert seen and set(seen)=={1e-12},f'Hidden unscreened direct call: {seen}'


def test_molecular_adapter_rejects_unmatched_local_xc_screening():
    # Local AO-XC screening and ERI quartet screening are distinct policies.
    # The new opt-in local-XC path has not yet defined exported response.
    mf=mean_field('RKS','pbe0','direct',grid_screen_tol=1e-14)
    with pytest.raises(NotImplementedError,match='(?i)(screened XC|grid_screen_tol)'):
        MolecularEnergy(mf,jk_backend='direct')


def test_molecular_adapter_inherits_direct_seed_representation():
    mf=mean_field('RHF','hf','direct')
    energy=MolecularEnergy(mf,forward='scf')
    assert energy.jk_backend=='direct'
    assert energy.scf_config.direct_scf_tol==mf.direct_scf_tol


def test_direct_geometry_workflow_inherits_seed_memory_budget():
    mf=mean_field('RHF','hf','direct')
    energy=MolecularEnergy(mf,forward='scf')
    assert energy.max_memory==mf.max_memory


def test_uks_factorized_none_eri_matches_empty_placeholder():
    from gradscf.scf.uks import UKSConfig, run_uks_from_integrals
    rng=np.random.default_rng(921)
    factors=jnp.asarray(rng.normal(size=(4,3,3))*.03)
    factors=(factors+factors.swapaxes(-1,-2))*.5
    kwargs=dict(overlap=jnp.eye(3),hcore=jnp.diag(jnp.array([-1.,-.4,.8])),
        nalpha=2,nbeta=1,nuclear_repulsion=0.,ao=jnp.zeros((0,3)),
        ao_deriv1=jnp.zeros((4,0,3)),grid_weights=jnp.zeros(0),df_factors=factors,
        config=UKSConfig(xc_spec='hf',jk_backend='df',conv_tol=1e-11,conv_tol_grad=1e-9))
    actual=run_uks_from_integrals(eri=None,**kwargs)
    expected=run_uks_from_integrals(eri=jnp.zeros((0,0,0,0)),**kwargs)
    assert actual.converged and expected.converged
    np.testing.assert_allclose(actual.total_energy,expected.total_energy,atol=1e-12,rtol=0)
    np.testing.assert_allclose(actual.density_matrix_alpha,expected.density_matrix_alpha,atol=1e-12,rtol=0)


def test_full_uks_rejects_missing_eri():
    from gradscf.scf.uks import UKSConfig, run_uks_from_integrals
    with pytest.raises(ValueError,match='Full-integral UKS requires an ERI'):
        run_uks_from_integrals(overlap=jnp.eye(2),hcore=jnp.eye(2),eri=None,
            nalpha=1,nbeta=1,nuclear_repulsion=0.,ao=jnp.zeros((0,2)),
            ao_deriv1=jnp.zeros((4,0,2)),grid_weights=jnp.zeros(0),
            config=UKSConfig(xc_spec='hf',jk_backend='full'))
