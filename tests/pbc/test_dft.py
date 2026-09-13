import numpy as np
import pytest
from gradscf.pbc import gto,dft


@pytest.mark.parametrize('nk',[1,2])
@pytest.mark.parametrize('xc',['svwn','pbe','pbe0'])
@pytest.mark.parametrize('unrestricted',[False,True])
def test_gamma_dft_matches_pyscf(xc,unrestricted,nk):
    pytest.importorskip('jax_xc');pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,dft as pdft
    kwargs=dict(atom='H 0 0 0; H 0 0 1.4',a=np.eye(3)*6,unit='Bohr',basis='gth-szv',pseudo='gth-pade',mesh=(31,)*3,precision=1e-10)
    cell=gto.M(**kwargs)
    kpts=cell.make_kpts((nk,1,1))
    cls=(dft.UKS if unrestricted else dft.RKS) if nk==1 else (dft.KUKS if unrestricted else dft.KRKS)
    mf=cls(cell,xc=xc,kpts=kpts).run()
    refcell=pgto.M(**kwargs,cart=True,verbose=0)
    cls=(pdft.UKS if unrestricted else pdft.RKS) if nk==1 else (pdft.KUKS if unrestricted else pdft.KRKS)
    ref=cls(refcell,**({} if nk==1 else {'kpts':np.asarray(kpts)}))
    ref.xc=xc;ref.conv_tol=1e-11;ref.conv_tol_grad=1e-7;ref.kernel()
    assert mf.converged and ref.converged
    np.testing.assert_allclose(mf.e_tot,ref.e_tot,atol=1e-6,rtol=0)


@pytest.mark.parametrize('nk',[1,2])
def test_spin_polarized_periodic_pbe(nk):
    pytest.importorskip('jax_xc');pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,dft as pdft
    kw=dict(atom='H 1 1 1',a=np.eye(3)*6,unit='Bohr',spin=1,mesh=(41,)*3,
            basis='gth-szv',pseudo='gth-pade')
    cell=gto.M(**kw);kpts=cell.make_kpts((nk,1,1))
    mf=(dft.UKS if nk==1 else dft.KUKS)(cell,kpts=kpts,xc='pbe').run()
    rc=pgto.M(**kw,cart=True,verbose=0)
    ref=(pdft.UKS(rc) if nk==1 else pdft.KUKS(rc,kpts=np.asarray(kpts)))
    if nk>1: ref.nelec=tuple(nk*n for n in cell.nelec)
    ref.xc='pbe';ref.conv_tol=1e-11;ref.kernel()
    assert mf.converged and ref.converged
    np.testing.assert_allclose(mf.e_tot,ref.e_tot,atol=1e-6,rtol=0)
