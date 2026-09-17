import numpy as np
import pytest
from gradscf.pbc import gto,scf
from gradscf.integrals.periodic.fft import build_inputs,get_kpoint_jk


def test_complex_kpoint_matrices_and_hf():
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,scf as pscf
    kw=dict(atom='H .2 .3 .4; H 1.6 .3 .4',a=np.eye(3)*6,unit='Bohr',basis='gth-szv',pseudo='gth-pade',mesh=(31,)*3,precision=1e-10)
    cell=gto.M(**kw);kpts=cell.make_kpts((2,1,1));inputs=build_inputs(cell,kpts=kpts)
    refcell=pgto.M(**kw,cart=True,verbose=0);ref=pscf.KRHF(refcell,kpts=np.asarray(kpts))
    np.testing.assert_allclose(inputs.overlap,ref.get_ovlp(),atol=1e-7,rtol=0)
    np.testing.assert_allclose(inputs.hcore,ref.get_hcore(),atol=2e-6,rtol=0)
    dm=np.broadcast_to(np.eye(2)*.3,(2,2,2,2)).copy()
    j,k=get_kpoint_jk(inputs,dm,mesh=cell.mesh)
    jr,kr=ref.get_jk(dm_kpts=dm)
    np.testing.assert_allclose(j,jr.sum(axis=0),atol=2e-6,rtol=0)
    np.testing.assert_allclose(k,kr,atol=2e-6,rtol=0)
    mf=scf.KRHF(cell,kpts=kpts).run()
    ref.conv_tol=1e-11;ref.kernel()
    assert mf.converged and ref.converged
    np.testing.assert_allclose(mf.e_tot,ref.e_tot,atol=1e-6,rtol=0)
