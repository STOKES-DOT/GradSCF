"""Opt-in local XC preserves small complete conventional/RI SCF tasks."""
import numpy as np
import pytest

from gradscf import dft
from test_hfks_acceptance import _molecule, _assert_stationary


@pytest.mark.parametrize('method',['RKS','UKS','GKS'])
@pytest.mark.parametrize('representation',['direct','df'])
def test_screened_public_scf_preserves_full_space_and_stationary_energy(method,representation):
    mol=_molecule(method)
    results=[]
    for cutoff in (0.,1e-14):
        mf=getattr(dft,method)(mol,xc='pbe0',grid_screen_tol=cutoff,
            grids_level=0,init_guess='hcore',conv_tol=1e-11,
            conv_tol_density=1e-9,conv_tol_grad=1e-8,max_cycle=150,max_memory=1024.,
            jk_backend=representation,direct_scf_workers=2,direct_scf_tol=1e-12)
        if representation=='df':mf=mf.density_fit(auxbasis='def2-universal-jkfit')
        mf.run();_assert_stationary(mf)
        results.append(mf)
        if cutoff and method=='UKS':assert mf.reference is None
    np.testing.assert_allclose(results[0].e_tot,results[1].e_tot,atol=1e-7,rtol=0)
    assert results[0].mo_coeff.shape==results[1].mo_coeff.shape
