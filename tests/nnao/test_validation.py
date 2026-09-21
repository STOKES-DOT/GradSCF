"""Saved NNAO endpoint summaries remain readable across backend key migration."""
import json
import runpy

import pytest


@pytest.mark.parametrize('key,value', [('eri_backend','full'),('eri_backend','ri'),
                                      ('jk_backend','direct'),('jk_backend','df')])
def test_endpoint_validator_accepts_old_and_new_backend_keys(tmp_path,key,value):
    pyscf=pytest.importorskip('pyscf')
    mol=pyscf.gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g',verbose=0)
    mf=mol.RHF()
    auxbasis='def2-universal-jkfit'
    if value in ('ri','df'):mf=mf.density_fit(auxbasis=auxbasis)
    mf.run(conv_tol=1e-12)
    assert mf.converged
    shells=[mol._basis['H'],mol._basis['H']]
    summary={key:value,'symbols':['H','H'],'coords_angstrom':[[0.,0.,0.],[0.,0.,.74]],
             'cartesian':False,'auxbasis':auxbasis,'initial_basis':shells,'final_basis':shells,
             'initial_energy_hartree':mf.e_tot,'final_energy_hartree':mf.e_tot}
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(summary))
    runpy.run_path('tests/comparisons/validate_methane_nnao.py')['validate'](path)
    result=json.loads((tmp_path/'pyscf-validation.json').read_text())
    assert all(v['absolute_error_hartree'] < 1e-8 for v in result.values())
