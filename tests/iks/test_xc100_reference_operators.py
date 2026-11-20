"""Opt-in diagnostic of the immutable operators used for orbital references."""
import json
import os
from pathlib import Path

import numpy as np
import pytest


ROOT=os.environ.get('GRADSCF_XC100_ACCEPTANCE')
pytestmark=pytest.mark.skipif(not ROOT,reason='Qualified XC100 reference data required')


def test_restored_reference_roundoff_is_not_a_model_failure():
    from gradscf.dft import iks
    from gradscf.model.autoxc.workflows.xc100.campaign import write_json
    root=Path(ROOT);rows=[];failed=[]
    for i in range(1,101):
        identifier=f'{i:03d}'
        with np.load(root/'data/xc100'/identifier/'density.npz',allow_pickle=False) as a:
            base=a['kinetic']+a['nuclear']+a['hartree']
        with np.load(root/'reference/cache/cases'/identifier/'pbe.npz',allow_pickle=False) as a:
            fock=base+a['potential']
        norm=float(np.linalg.norm(fock));anti=fock-fock.T
        old=32*np.finfo(fock.dtype).eps*max(1.,norm)
        row={'system_id':identifier,'fock_norm':norm,'antisymmetric_norm':float(np.linalg.norm(anti)),
             'antisymmetric_max':float(np.max(np.abs(anti))),'old_threshold':old,
             'finite':bool(np.isfinite(fock).all())}
        rows.append(row)
        if row['antisymmetric_norm']>old:failed.append((identifier,fock))
    write_json(root/'validation/reference-operator-diagnostic.json',{'rows':rows,
        'old_rejected_ids':[i for i,_ in failed]})
    assert all(r['finite'] and r['antisymmetric_max']<=1e-10 for r in rows)
    assert failed,'Diagnostic must exercise the observed restored-operator discrepancy'
    identifier,fock=failed[0]
    with np.load(root/'data/xc100'/identifier/'density.npz',allow_pickle=False) as a:
        density,overlap=a['density'],a['overlap']
    metadata=json.loads((root/'data/xc100'/identifier/'metadata.json').read_text())
    target=iks.Target(density,overlap,metadata['nelec'],occupation_tolerance=1e-8)
    reference=iks.prepare_orbital_reference(target,fock,method='restored-PBE@target')
    assert reference.valid
