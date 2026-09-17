import numpy as np
import pytest
from gradscf.pbc import gto,scf,dft,tdscf


@pytest.mark.parametrize('nk',[1,2,3])
@pytest.mark.parametrize('tda',[True,False])
def test_periodic_hf_excitations_match_reference(nk,tda):
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,scf as pscf,tdscf as ptd
    kw=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6,unit='Bohr',basis='gth-szv',pseudo='gth-pade',mesh=(41,)*3,precision=1e-10)
    cell=gto.M(**kw);kpts=cell.make_kpts((nk,1,1))
    mf=(scf.RHF(cell) if nk==1 else scf.KRHF(cell,kpts=kpts)).run()
    rc=pgto.M(**kw,cart=True,verbose=0)
    ref=(pscf.RHF(rc) if nk==1 else pscf.KRHF(rc,kpts=np.asarray(kpts)))
    ref.conv_tol=1e-11;ref.kernel()
    ours=(tdscf.TDA if tda else tdscf.TDHF)(mf,nstates=1)
    e,_=ours.kernel()
    reference=(ptd.TDA if tda else ptd.TDHF)(ref)
    reference.nstates=1;reference.conv_tol=1e-8;reference.kernel()
    assert bool(np.all(ours.converged))
    np.testing.assert_allclose(e,np.asarray(reference.e).reshape(-1)[:1],atol=1e-5,rtol=0)


@pytest.mark.parametrize('nk',[1,2])
@pytest.mark.parametrize('tda',[True,False])
@pytest.mark.parametrize('xc',['pbe','pbe0'])
def test_periodic_dft_excitations_match_reference(nk,tda,xc):
    pytest.importorskip('jax_xc');pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,dft as pdft,tdscf as ptd
    kw=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6,unit='Bohr',basis='gth-szv',pseudo='gth-pade',mesh=(41,)*3,precision=1e-10)
    cell=gto.M(**kw);kpts=cell.make_kpts((nk,1,1))
    mf=(dft.RKS if nk==1 else dft.KRKS)(cell,kpts=kpts,xc=xc).run()
    rc=pgto.M(**kw,cart=True,verbose=0)
    ref=pdft.RKS(rc) if nk==1 else pdft.KRKS(rc,kpts=np.asarray(kpts))
    ref.xc=xc;ref.conv_tol=1e-11;ref.kernel()
    ours=(tdscf.TDA if tda else tdscf.TDDFT)(mf,nstates=1)
    e,_=ours.kernel()
    reference=(ptd.TDA if tda else ptd.TDDFT)(ref)
    reference.nstates=1;reference.conv_tol=1e-8;reference.kernel()
    assert bool(np.all(ours.converged))
    np.testing.assert_allclose(e,np.asarray(reference.e).reshape(-1)[:1],atol=1e-5,rtol=0)


@pytest.mark.parametrize('tda',[True,False])
def test_unrestricted_gamma_response(tda):
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,scf as pscf,tdscf as ptd
    kw=dict(atom='H 0 0 0; H 0 0 1.4',a=np.eye(3)*6,unit='Bohr',
            basis='gth-szv',pseudo='gth-pade',mesh=(41,)*3)
    mf=scf.UHF(gto.M(**kw)).run()
    ref=pscf.UHF(pgto.M(**kw,cart=True,verbose=0));ref.conv_tol=1e-11;ref.kernel()
    ours=(tdscf.TDA if tda else tdscf.TDHF)(mf,nstates=1).run()
    reference=(ptd.TDA if tda else ptd.TDHF)(ref)
    reference.nstates=1;reference.conv_tol=1e-8;reference.kernel()
    assert bool(np.all(ours.converged))
    np.testing.assert_allclose(ours.e,np.asarray(reference.e).reshape(-1)[:1],atol=1e-5,rtol=0)


@pytest.mark.parametrize('change',['geometry','mesh'])
def test_cached_response_rejects_changed_cell_until_scf_reruns(change):
    cell=gto.M(atom='H 0 0 0; H 0 0 1.4',a=np.eye(3)*6,unit='Bohr',
               basis='gth-szv',pseudo='gth-pade',mesh=(15,)*3)
    mf=scf.RHF(cell).run()
    td=tdscf.TDA(mf,nstates=1).run()
    previous=np.asarray(td.e).copy()
    assert np.all(td.converged)
    if change=='geometry':
        cell.atom='H 0 0 0; H 0 0 2.0'
        cell.build()
    else:
        cell.mesh=(17,)*3
    for evaluate in (td.kernel,td.get_ab):
        with pytest.raises(ValueError,match='Run kernel.*periodic reference'):
            evaluate()
    mf.kernel()
    td.kernel()
    assert np.all(td.converged)
    fresh=tdscf.TDA(mf,nstates=1).run()
    np.testing.assert_allclose(td.e,fresh.e,atol=1e-10,rtol=0)
    if change=='geometry':
        assert np.max(np.abs(np.asarray(td.e)-previous))>1e-3
