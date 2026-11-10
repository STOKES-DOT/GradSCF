"""Teacher batches may launch only from verified frozen raw inputs."""
import hashlib
import json

import pytest


def inputs(tmp_path):
    root = tmp_path/'inputs'
    (root/'raw_subset').mkdir(parents=True)
    (root/'qualification').mkdir()
    rows = []
    ids = [f'm{i:02d}' for i in range(30)]
    for name in ids:
        p = root/'raw_subset'/f'{name}.npz'
        p.write_bytes(name.encode())
        p.with_suffix('.metadata.json').write_text('{}')
        rows.append(dict(molecule_id=name, sample_id=name+':frame-1', npz=p.name,
            npz_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), qualification={'status':'qualified'}))
    payload = dict(status='qualified', molecule_ids=ids, rows=rows,
                   local_file_provenance={'archive_sha256_verified':True})
    (root/'qualification/qualification.json').write_text(json.dumps(payload))
    return root


def test_planner_preserves30_declared_molecules_and_checks_raw_hashes(tmp_path):
    from gradscf_tools.iks_workflow import plan_teacher_jobs
    root = inputs(tmp_path)
    jobs = plan_teacher_jobs(root)
    assert len(jobs) == 30 and jobs[0]['molecule_id'] == 'm00'
    (root/'raw_subset/m00.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash|SHA'):
        plan_teacher_jobs(root)


def test_unverified_or_incomplete_source_cannot_launch_teacher_batch(tmp_path):
    from gradscf_tools.iks_workflow import plan_teacher_jobs
    root = inputs(tmp_path)
    path = root/'qualification/qualification.json'
    payload = json.loads(path.read_text())
    payload['local_file_provenance']['archive_sha256_verified'] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='verified|archive'):
        plan_teacher_jobs(root)


def reference_manifest():
    names = [f'm{i:02d}' for i in range(30)]
    split = {'train':names[:20], 'validation':names[20:25], 'test':names[25:]}
    rows = [dict(molecule_id=name, split=role, valid=True)
            for role,ids in split.items() for name in ids]
    return dict(status='qualified', protocol_sha256='protocol-hash', split=split, rows=rows)


def test_training_validation_selection_never_opens_sealed_test():
    from gradscf_tools.iks_workflow import select_references
    manifest = reference_manifest()
    rows = select_references(manifest)
    assert len(rows) == 25 and all(row['split'] != 'test' for row in rows)
    with pytest.raises(ValueError, match='sealed|frozen'):
        select_references(manifest, splits=('test',))
    frozen = {'status':'frozen', 'candidate_id':'c001', 'protocol_sha256':'protocol-hash'}
    assert len(select_references(manifest, splits=('test',), frozen_winner=frozen)) == 5


def test_mislabeled_or_failed_reference_cannot_enter_fitting():
    from gradscf_tools.iks_workflow import select_references
    manifest = reference_manifest()
    manifest['rows'][-1]['split'] = 'train'
    with pytest.raises(ValueError, match='split|role'):
        select_references(manifest)


def test_batch_dispatches_declared_inverse_method_and_regularization(tmp_path, monkeypatch):
    from gradscf_tools import iks_workflow as workflow
    from gradscf_tools.pbe_recording import canonical_hash
    root = inputs(tmp_path)
    ids = [f'm{i:02d}' for i in range(30)]
    split = {'train':ids[:20], 'validation':ids[20:25], 'test':ids[25:]}
    (root/'qualification/sample_manifest.json').write_text(json.dumps({'split':split}))
    protocol = dict(method='density_match', grid_level=1, density_metric='relative_l2',
        regularization=1e-10, gradient_max=1e-8, density_tolerance=1e-3)
    protocol['sha256'] = canonical_hash(protocol)
    protocol_path = tmp_path/'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    source = tmp_path/'source'
    (source/'tools').mkdir(parents=True)
    (source/'tools/cafes_wuyang_teacher.py').write_text('# unique test generator')
    seeds = tmp_path/'seeds'
    for name in ids:
        (seeds/name).mkdir(parents=True)
        (seeds/name/'teacher.json').write_text(json.dumps({'gap':.1}))
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        def arg(flag):
            return command[command.index(flag)+1]
        assert arg('--method') == 'density_match'
        assert float(arg('--regularization')) == 1e-10
        assert Path(arg('--initial-teacher')).parent.name == Path(arg('--output')).parent.name
        quality = dict(regularization=1e-10, gradient_tolerance=1e-8,
            diagnostic_tolerance=1e-3, density_error_kind='relative_l2', inverse_method='density_match')
        Path(arg('--output')).write_text(json.dumps(dict(valid=True, inverse_method='density_match',
            grid_level=1, quality_settings=quality)))
        return type('Process', (), {'returncode':0})()
    from pathlib import Path
    monkeypatch.setattr(workflow.subprocess, 'run', execute)
    result = workflow.run_teacher_batch(root, tmp_path/'results', protocol_path, source,
                                        seed_root=seeds)
    assert result['status'] == 'qualified' and len(calls) == 30
    assert all(row['valid'] for row in result['rows'])
