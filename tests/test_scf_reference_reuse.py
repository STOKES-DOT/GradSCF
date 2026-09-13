"""Preparing response data must reuse the completed RKS calculation."""
from dataclasses import replace
import numpy as np
import pytest

from gradscf import gto, scf
from gradscf.scf import facade, builders


@pytest.mark.parametrize('backend', ['full','df'])
def test_rks_reference_reuses_integrals_and_scf_state(monkeypatch, backend):
    calls={'integrals':0,'scf':0}
    build,run=facade.build_rks_integral_inputs,facade.run_rks_from_integrals
    def counted_build(**kwargs):
        calls['integrals']+=1
        return build(**kwargs)
    def counted_run(**kwargs):
        calls['scf']+=1
        return run(**kwargs)
    monkeypatch.setattr(facade,'build_rks_integral_inputs',counted_build)
    monkeypatch.setattr(facade,'run_rks_from_integrals',counted_run)
    monkeypatch.setattr(builders,'build_rks_integral_inputs',counted_build)
    monkeypatch.setattr(builders,'run_rks_from_integrals',counted_run)
    mf=scf.RKS(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g'),xc='hf',jk_backend=backend)
    energy=mf.kernel()
    coeff=np.asarray(mf.mo_coeff).copy()
    result=mf.scf_result
    reference=mf._ensure_reference()
    assert calls=={'integrals':1,'scf':1}
    assert mf._ensure_reference() is reference
    assert mf.scf_result is result
    np.testing.assert_allclose(reference.rdm1.sum(axis=0),result.density_matrix,atol=0,rtol=0)
    np.testing.assert_allclose(mf.mo_coeff,coeff,atol=0,rtol=0)
    assert float(reference.mf_energy)==float(energy)
    assert reference.dipole_integrals.shape==(3,2,2)
    assert mf.cycles==result.cycles and mf.converged==result.converged
    if backend=='df':
        assert reference.df_factors is not None
    td=mf.TDA()
    td.nstates=1
    td.kernel()
    assert calls=={'integrals':1,'scf':1}
    mf.kernel()
    assert calls=={'integrals':2,'scf':2}
    assert mf.reference is None
    assert mf._ensure_reference() is not reference
    assert calls=={'integrals':2,'scf':2}


def test_rks_reference_rejects_changed_geometry_or_scf_configuration():
    mf=scf.RKS(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g'),xc='hf').run()
    mf._ensure_reference()
    mf.conv_tol_density=1e-5
    with pytest.raises(RuntimeError,match='kernel'):
        mf._ensure_reference()
    with pytest.raises(RuntimeError,match='kernel'):
        mf.TDA().kernel()
    mf.kernel()
    mf.mol=replace(mf.mol,atom='H 0 0 0; H 0 0 .9')
    with pytest.raises(RuntimeError,match='kernel'):
        mf._ensure_reference()


def test_rks_facade_forwards_independent_gradient_tolerance():
    mf=scf.RKS(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g'),conv_tol_grad=3e-9)
    assert mf._config().conv_tol_grad==3e-9


def test_response_feature_changes_reuse_the_same_ground_state(monkeypatch):
    mf=scf.RKS(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g'),xc='hf').run()
    before=mf._ensure_reference()
    result=mf.scf_result
    def forbidden(**kwargs):
        raise AssertionError('Reference feature changes must not run SCF or rebuild inputs')
    monkeypatch.setattr(builders,'run_rks_from_integrals',forbidden)
    monkeypatch.setattr(builders,'build_rks_integral_inputs',forbidden)
    calls=[]
    def extra(basis,ao,densities,coords,**kwargs):
        calls.append(kwargs)
        np.testing.assert_allclose(densities[0]*2,result.density_matrix,atol=0,rtol=0)
        shape=(2,len(kwargs['omega_values']),len(coords))
        return np.zeros(shape),np.zeros(shape)
    monkeypatch.setattr(builders,'_local_hfx_features_from_basis_dm',extra)
    mf.compute_local_hfx_features=True
    after=mf._ensure_reference()
    assert after is not before and len(calls)==1
    assert mf.scf_result is result
    assert mf._ensure_reference() is after
