import numpy as np
import pytest
from gradscf.pbc import gto,scf


@pytest.mark.parametrize('unrestricted',[False,True])
def test_gamma_hf_matches_pyscf(unrestricted):
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,scf as pscf
    kwargs=dict(atom='H 0 0 0; H 0 0 1.4',a=np.eye(3)*6,unit='Bohr',basis='gth-szv',pseudo='gth-pade',mesh=(31,)*3,precision=1e-10)
    cell=gto.M(**kwargs)
    cls=scf.UHF if unrestricted else scf.RHF
    mf=cls(cell).run()
    refcell=pgto.M(**kwargs,cart=True,verbose=0)
    ref=(pscf.UHF if unrestricted else pscf.RHF)(refcell)
    ref.conv_tol=1e-11;ref.conv_tol_grad=1e-7;ref.kernel()
    assert mf.converged and ref.converged
    np.testing.assert_allclose(mf.e_tot,ref.e_tot,atol=1e-6,rtol=0)
    density=mf.make_rdm1()
    expected=ref.make_rdm1()
    np.testing.assert_allclose(density[:,0] if unrestricted else density[0],expected,atol=2e-6,rtol=0)


def test_silicon_diamond_primitive_cell_hf():
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,scf as pscf
    a=np.array([[0,5.13,5.13],[5.13,0,5.13],[5.13,5.13,0]])
    kw=dict(atom='Si 0 0 0; Si 2.565 2.565 2.565',a=a,unit='Bohr',
            basis='gth-szv',pseudo='gth-pade',mesh=(31,)*3,precision=1e-10)
    mf=scf.RHF(gto.M(**kw)).run()
    ref=pscf.RHF(pgto.M(**kw,cart=True,verbose=0));ref.conv_tol=1e-11;ref.kernel()
    assert mf.converged and ref.converged
    np.testing.assert_allclose(mf.e_tot,ref.e_tot,atol=1e-6,rtol=0)
