from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc import optics


def test_velocity_oscillator_strength_normalization(monkeypatch):
    matrix=jnp.zeros((3,2,2),dtype=complex).at[0,0,1].set(1j).at[0,1,0].set(-1j)
    monkeypatch.setattr(optics,'velocity_matrix',lambda cell:matrix)
    result=SimpleNamespace(mo_coeff_spin=jnp.stack([jnp.eye(2)[None]]*2))
    cell=SimpleNamespace(nelec=(1,1),_version=1,mesh=(3,3,3))
    mf=SimpleNamespace(kpts=jnp.zeros((1,3)),unrestricted=False,result=result,cell=cell,
        xc='pbe',_computed_xc='pbe',_computed_mesh=cell.mesh,_cell_version=1,inputs=object())
    td=SimpleNamespace(_scf=mf,singlet=True,e=jnp.array([2.]),xy=(jnp.array([[2.]]),jnp.zeros((1,1))),
        converged=jnp.array([True]))
    td._solution_reference=(id(result),True,'pbe',id(mf.inputs))
    strength=optics.oscillator_strength(td)
    np.testing.assert_allclose(strength,[2/3],atol=1e-12,rtol=0)
    grid=jnp.linspace(45,65,4001)
    spectrum=optics.broaden_spectrum(td.e,strength,grid,fwhm_ev=.3)
    np.testing.assert_allclose(jnp.trapezoid(spectrum,grid),2/3,atol=1e-8,rtol=0)


def test_gth_velocity_matches_reference_operator():
    pytest.importorskip('pyscf.pbc.gto.pseudo.ppnl_velgauge')
    from pyscf.pbc import gto as pgto
    from pyscf.pbc.gto.pseudo.ppnl_velgauge import get_gth_pp_nl_velgauge_commutator
    from gradscf.pbc import gto
    kw=dict(atom='Si 0 0 0; Si 2.565 2.565 2.565',
        a=np.array([[0,5.13,5.13],[5.13,0,5.13],[5.13,5.13,0]]),unit='Bohr',
        basis='gth-szv',pseudo='gth-pbe',mesh=(41,)*3,precision=1e-10)
    cell=gto.M(**kw);reference=pgto.M(**kw,cart=True,verbose=0)
    momentum=reference.pbc_intor('int1e_ipovlp',comp=3,hermi=0,kpts=np.zeros(3))
    correction=get_gth_pp_nl_velgauge_commutator(reference,q=np.zeros(3),kpts=np.zeros(3))
    actual=optics.velocity_matrix(cell)
    assert np.linalg.norm(correction)>1e-3
    np.testing.assert_allclose(actual,1j*(momentum-correction),atol=2e-6,rtol=0)
    np.testing.assert_allclose(actual,actual.conj().swapaxes(-1,-2),atol=1e-10,rtol=0)
