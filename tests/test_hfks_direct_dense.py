"""Small independent native direct/dense SCF parity for all six facades."""
import numpy as np
import pytest
from gradscf import gto,scf,dft

@pytest.mark.parametrize('method',['RHF','UHF','GHF','RKS','UKS','GKS'])
def test_direct_dense_scf(method):
    opened=method[0] in 'UG'
    mol=gto.M(atom='Li 0 0 0; H 0 0 1.6',basis='sto-3g',cart=True,
              charge=1 if opened else 0,spin=1 if opened else 0)
    constructor=getattr(dft if method.endswith('KS') else scf,method)
    options=dict(conv_tol=1e-10,max_cycle=100,max_memory=1024.)
    if method.endswith('KS'):options['xc']='pbe'
    dense=constructor(mol,**options)
    direct=constructor(mol,**options).direct_scf()
    direct.direct_scf_tol=1e-12;direct.direct_scf_workers=2
    values=[mf.kernel() for mf in (dense,direct)]
    assert dense.converged and direct.converged
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[1],values[0],atol=1e-8,rtol=0)
    np.testing.assert_allclose(direct.make_rdm1(),dense.make_rdm1(),atol=2e-7,rtol=0)
