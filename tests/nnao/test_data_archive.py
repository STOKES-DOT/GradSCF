"""Scientific release exports must preserve inputs and discard optimizer state."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


TOOLS=Path(__file__).resolve().parents[2]/'reproducibility/nnao/tools'


def module(name):
    spec=importlib.util.spec_from_file_location(name,TOOLS/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_joint_export_preserves_theta_and_beta_but_no_optimizer(tmp_path):
    source=tmp_path/'last.npz'
    theta=np.array([.2,-.4]);beta=np.array([.1])
    np.savez(source,theta=theta,beta=beta,optimizer=np.ones(5),gradient=np.zeros(2),
        metadata=json.dumps({'identity':{'exponent_keys':['C:s']}}))
    output=tmp_path/'output'
    module('export_inference').export([{'path':str(source)}],output)
    report=json.loads((output/'index.json').read_text())
    assert not report['errors'] and len(report['records'])==1
    with np.load(output/report['records'][0]['parameters_path']) as saved:
        assert set(saved.files)=={'theta','beta'}
        np.testing.assert_array_equal(saved['theta'],theta)
        np.testing.assert_array_equal(saved['beta'],beta)


def test_equal_final_parameter_vectors_share_one_payload(tmp_path):
    sources=[]
    for name in ('best','last'):
        p=tmp_path/(name+'.npz')
        np.savez(p,parameters=np.array([1.,2.]),optimizer=np.ones(7),
            gradient=np.ones(2),metadata=json.dumps({'identity':{'model_config':{'channels':8}}}))
        sources.append({'path':str(p)})
    output=tmp_path/'output'
    module('export_inference').export(sources,output)
    report=json.loads((output/'index.json').read_text())
    assert not report['errors'] and report['unique_parameter_vectors']==1
    assert len(report['records'])==2
    with np.load(output/report['records'][0]['parameters_path']) as saved:
        assert saved.files==['parameters']


def test_collection_excludes_integrals_and_optimizer_but_keeps_density(tmp_path):
    source=tmp_path/'source';source.mkdir()
    np.savez(source/'integrals.npz',overlap=np.eye(2),hcore=np.eye(2),df_factors=np.ones((1,3)))
    np.savez(source/'last.npz',parameters=np.ones(2),optimizer=np.ones(2))
    np.savez(source/'density.npz',density=np.eye(2))
    (source/'references.json').write_text('{"structures":[]}')
    output=tmp_path/'output'
    module('collect_data').collect({'sources':[{'path':str(source),'destination':'data'}]},output)
    assert (output/'data/density.npz').exists()
    assert (output/'data/references.json').read_bytes()==(source/'references.json').read_bytes()
    assert not (output/'data/last.npz').exists() and not (output/'data/integrals.npz').exists()


def test_collection_cannot_overwrite_different_scientific_record(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'energies.csv').write_text('energy\n-1\n')
    plan={'sources':[{'path':str(source),'destination':'data'}]};output=tmp_path/'output'
    collector=module('collect_data');collector.collect(plan,output)
    (source/'energies.csv').write_text('energy\n-2\n')
    with pytest.raises(RuntimeError,match='differs'):
        collector.collect(plan,output)


def test_legacy_adam_state_is_recognized_as_training_state(tmp_path):
    path=tmp_path/'checkpoint.npz'
    np.savez(path,parameters=np.ones(2),adam_state=np.zeros(7),step=np.array(3))
    assert module('collect_data').excluded_npz(path)=='optimizer_or_training_checkpoint'


def test_optimizer_stripping_preserves_scientific_arrays_and_source_identity(tmp_path):
    root=tmp_path/'data';root.mkdir();p=root/'checkpoint.npz'
    parameters=np.array([-.0,2.]);outputs=np.array([[1.,np.nan]])
    np.savez(p,parameters=parameters,outputs=outputs,adam_state=np.arange(7),step=np.array(4))
    import hashlib
    original=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'files':[{'path':p.name,'sha256':original,'bytes':p.stat().st_size,
        'source_sha256':original,'source_bytes':p.stat().st_size}]}
    (root/'manifest.json').write_text(json.dumps(manifest))
    quarantine=tmp_path/'private'
    changes=module('strip_optimizer_arrays').strip(root,'manifest.json',quarantine)
    assert len(changes)==1 and changes[0]['removed_keys']==['adam_state']
    assert hashlib.sha256((quarantine/p.name).read_bytes()).hexdigest()==original
    with np.load(p) as saved:
        assert set(saved.files)=={'parameters','outputs','step'}
        assert saved['parameters'].tobytes()==parameters.tobytes()
        assert saved['outputs'].tobytes()==outputs.tobytes()
    row=json.loads((root/'manifest.json').read_text())['files'][0]
    assert row['source_sha256']==original and row['sha256']!=original
