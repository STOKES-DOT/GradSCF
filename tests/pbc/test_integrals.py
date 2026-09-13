import numpy as np
import pytest
from gradscf.pbc import gto
from gradscf.integrals.periodic.fft import build_inputs,get_jk


@pytest.mark.parametrize('atom',['He 0 0 0','C 0 0 0','Si 0 0 0'])
def test_gamma_fft_inputs_and_jk_match_reference(atom):
    pytest.importorskip('pyscf')
    from pyscf.pbc import gto as pgto,scf as pscf
    kwargs=dict(atom=atom,a=np.eye(3)*5,unit='Bohr',basis='gth-szv',pseudo='gth-pade',mesh=(41,41,41),precision=1e-10)
    cell=gto.M(**kwargs);inputs=build_inputs(cell)
    refcell=pgto.M(**kwargs,cart=True,verbose=0)
    mf=pscf.RHF(refcell)
    np.testing.assert_allclose(inputs.nuclear_repulsion,refcell.energy_nuc(),atol=1e-8,rtol=0)
    np.testing.assert_allclose(inputs.overlap[0],mf.get_ovlp(),atol=1e-7,rtol=0)
    np.testing.assert_allclose(inputs.hcore[0],mf.get_hcore(),atol=2e-6,rtol=0)
    n=cell.topology.nao;dm=np.stack([np.eye(n)*.3,np.eye(n)*.2])
    j,k=get_jk(inputs,dm)
    jr,kr=mf.get_jk(dm=dm)
    np.testing.assert_allclose(j,jr.sum(axis=0),atol=2e-6,rtol=0)
    np.testing.assert_allclose(k,kr,atol=2e-6,rtol=0)
