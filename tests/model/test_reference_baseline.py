"""Imported reference operators retain their original data/model provenance."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def snapshot(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.molecule import file_hash
    import gradscf.model.autoxc.workflows.xc100.pbe as pbe
    folder=tmp_path/'reference/cache/cases/010';folder.mkdir(parents=True)
    path=folder/'pbe.npz'
    np.savez(path,energy=.3,potential=np.eye(2),core='original-source',fingerprint='original-case')
    model=SimpleNamespace(system_id='010',archive_hash='data',metadata_hash='metadata',
        metadata={'canonical_basis_sha256':'basis'},floor=1e-12,
        target=SimpleNamespace(density=np.eye(2)))
    receipt={'origin_core':'original-source','floor':1e-12,'pbe_model_sha256':file_hash(pbe.__file__),
        'cases':{'010':{'archive_sha256':'data','metadata_sha256':'metadata',
            'basis_sha256':'basis','cache_sha256':file_hash(path),'fingerprint':'original-case'}}}
    (tmp_path/'reference_baselines.json').write_text(json.dumps(receipt))
    return model,path


def test_explicit_reference_baseline_is_loaded_without_relabeling_its_origin(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.reference import reference_baseline
    model,_=snapshot(tmp_path)
    energy,potential=reference_baseline(tmp_path,model)
    assert energy==.3
    np.testing.assert_array_equal(potential,np.eye(2))


def test_reference_cache_or_target_changes_are_rejected(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.reference import reference_baseline
    model,path=snapshot(tmp_path)
    model.metadata_hash='different'
    with pytest.raises(ValueError): reference_baseline(tmp_path,model)
    model.metadata_hash='metadata'
    with path.open('ab') as handle:handle.write(b'tampered')
    with pytest.raises(ValueError): reference_baseline(tmp_path,model)


def test_strict_iks_reference_loader_does_not_require_an_energy_array(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.reference import reference_baseline
    from gradscf.model.autoxc.workflows.xc100.molecule import file_hash
    model,path=snapshot(tmp_path)
    np.savez(path,potential=np.eye(2),core='original-source',fingerprint='original-case')
    receipt=json.loads((tmp_path/'reference_baselines.json').read_text())
    receipt['cases']['010']['cache_sha256']=file_hash(path)
    (tmp_path/'reference_baselines.json').write_text(json.dumps(receipt))
    energy,potential=reference_baseline(tmp_path,model,include_energy=False)
    assert energy is None
    np.testing.assert_array_equal(potential,np.eye(2))
