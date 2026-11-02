"""Migration guards use NumPy only; no CUDA or electronic-structure runtime."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def api():
    path = Path(__file__).with_name('recover_train.py')
    assert path.exists(), 'Explicit checkpoint migration is not implemented.'
    spec = importlib.util.spec_from_file_location('recover_train_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identities(api):
    old = {name: 'old-' + name for name in api.IDENTITY_CHANGES}
    new = {name: 'new-' + name for name in api.IDENTITY_CHANGES}
    for identity in (old, new):
        identity.update(implicit_tolerance=1e-8, cache_storage='mmap',
                        structure_ids=['a', 'b'], learning_rate_decay_updates=1000)
    return old, new


def runtime():
    return dict(python_version='3.11.15', parameter_dtype='float64', versions={
        'jax': '0.10.0', 'jaxlib': '0.10.0', 'flax': '0.12.7',
        'optax': '0.2.8', 'e3nn-jax': '0.21.0', 'mace-jax': '0.2.0'})


@pytest.mark.parametrize('field', ['versions', 'python_version', 'parameter_dtype'])
def test_runtime_migration_rejects_environment_drift(api, field):
    parent = runtime(); current = runtime()
    assert api.verify_runtime_environment(parent, current) == parent
    if field == 'versions': current[field]['jax'] = '0.10.1'
    else: current[field] = 'changed'
    with pytest.raises(ValueError, match='environment'):
        api.verify_runtime_environment(parent, current)


def checkpoint():
    return dict(parameters=np.array([1., 2.]), gradient=np.array([.01, -.02]),
                epoch=41, loss_hartree=-1.5, molecules=[
                    dict(structure_id='a', energy_hartree=-1.),
                    dict(structure_id='b', energy_hartree=-2.)])


def test_only_explicit_source_and_gate_identity_fields_may_change(api):
    old, new = identities(api)
    assert set(api.verify_identity_transition(old, new)) == api.IDENTITY_CHANGES
    new['implicit_tolerance'] = 1e-7
    with pytest.raises(ValueError, match='identity'):
        api.verify_identity_transition(old, new)


def test_remeasurement_accepts_same_point_energy_and_gradient(api):
    saved = checkpoint()
    result = api.compatibility_metrics(saved, -1.5, saved['gradient'] + 1e-10,
                                       saved['molecules'])
    assert result['max_structure_energy_error_hartree'] == 0.
    assert result['gradient_passed'] is True


@pytest.mark.parametrize('failure', ['order', 'energy', 'mean', 'gradient', 'nan'])
def test_remeasurement_rejects_incompatible_or_nonfinite_batch(api, failure):
    saved = checkpoint(); records = list(saved['molecules'])
    gradient = saved['gradient'].copy(); loss = -1.5
    if failure == 'order': records = records[::-1]
    if failure == 'energy': records[0] = dict(records[0], energy_hartree=-1.0001)
    if failure == 'mean': loss += 2e-9
    if failure == 'gradient': gradient[0] += 1e-3
    if failure == 'nan': gradient[0] = np.nan
    with pytest.raises(ValueError):
        api.compatibility_metrics(saved, loss, gradient, records)


def test_partial_checkpoint_publication_never_becomes_normal_resume(api, tmp_path):
    assert api.output_state(tmp_path) == 'fresh'
    (tmp_path / 'last.npz').write_bytes(b'partial')
    with pytest.raises(ValueError, match='incomplete'):
        api.output_state(tmp_path)
    (tmp_path / 'migration.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='incomplete'):
        api.output_state(tmp_path)


def test_completed_migration_requires_all_three_checkpoints(api, tmp_path):
    (tmp_path / 'migration.json').write_text(json.dumps({'status': 'complete'}))
    with pytest.raises(ValueError, match='incomplete'):
        api.output_state(tmp_path)
    for name in ('last.npz', 'best.npz', 'best_validation.npz'):
        (tmp_path / name).write_bytes(b'published')
    assert api.output_state(tmp_path) == 'resume'


def test_optimizer_global_counters_must_remain_at_parent_epoch(api):
    state = dict(count=np.asarray(41), m=np.array([.1]), nested={'count': np.asarray(41)})
    assert api.verify_optimizer_counts(state, 41) == [41, 41]
    state['nested']['count'] = np.asarray(0)
    with pytest.raises(ValueError, match='counter'):
        api.verify_optimizer_counts(state, 41)


def test_regression_requires_original_checkpoint_and_complete_repetitions(api):
    report = dict(status='complete', checkpoint_sha256=api.PARENT_CHECKPOINT,
        original_checkpoint_bytes_unchanged=True, target_repetitions=100,
        slurm_job_id='94841', backend='gpu', provenance={'proof': 1},
        basis_directional_check={'passed': True}, parameter_directional_check={'passed': True},
        runs=[dict(structure_id='ADIM6/AD6', repeat=i, status='passed') for i in range(1, 101)])
    api.verify_regression(report, {'proof': 1})
    report['runs'].pop()
    with pytest.raises(ValueError, match='regression'):
        api.verify_regression(report, {'proof': 1})


def test_payload_refresh_preserves_parameters_optimizer_and_original_history(api):
    old, new = identities(api)
    metadata = dict(checkpoint(), identity=old, history=[{'epoch': 0}, {'epoch': 41}])
    metadata.pop('parameters'); metadata.pop('gradient')
    arrays = dict(parameters=np.array([1., 2.]), gradient=np.array([.01, -.02]),
                  optimizer=np.array([1, 2, 255], dtype=np.uint8))
    payload = api.refreshed_payload(metadata, arrays, new, {'parent': 'sha'},
                                    loss=-1.5, gradient=np.array([.02, -.01]),
                                    records=metadata['molecules'])
    np.testing.assert_array_equal(payload[1]['parameters'], arrays['parameters'])
    np.testing.assert_array_equal(payload[1]['optimizer'], arrays['optimizer'])
    np.testing.assert_array_equal(payload[1]['gradient'], [.02, -.01])
    assert payload[0]['history'] == metadata['history']
    assert metadata['identity'] == old
    assert payload[0]['identity'] == new and payload[0]['epoch'] == 41


def test_wrapper_forwards_formal_resume_through_the_standard_main(api, tmp_path, monkeypatch):
    seen = []
    trainer = SimpleNamespace(run_training=lambda *a, **kw: None,
        main=lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(api, 'import_path', lambda *args: trainer)
    result = api.main(['--parent-run', str(tmp_path / 'old'),
        '--parent-manifest', 'parent.json', '--new-manifest', 'new.json',
        '--regression', 'regression.json', '--', '--output-dir', str(tmp_path / 'new'), '--resume'])
    assert result == 0 and '--resume' in seen[0]


def test_ordinary_resume_rejects_native_source_drift(api, tmp_path, monkeypatch):
    def changed_source(*args):
        raise RuntimeError('Live repaired file differs from its frozen manifest: native/example.cc')
    monkeypatch.setattr(api, 'import_path', lambda *args:
        SimpleNamespace(verify_sources=changed_source))
    args = SimpleNamespace(parent_manifest=tmp_path / 'parent.json',
                           new_manifest=tmp_path / 'new.json')
    marker = {'source_transition': {'parent_identity': {'scientific_source_sha256': api.PARENT_SOURCE},
                                    'provenance': {'verified_file_count': 764}}}
    with pytest.raises(RuntimeError, match='native/example.cc'):
        api.verify_resume_sources(args, marker)


@pytest.fixture
def migration_case(api, tmp_path, monkeypatch):
    parent = tmp_path / 'parent'; parent.mkdir()
    output = tmp_path / 'new'; output.mkdir()
    old, new = identities(api)
    for identity in (old, new):
        identity['structure_ids'] = [f'train/{i}' for i in range(200)]
        identity['heldout_structure_ids'] = [f'heldout/{i}' for i in range(50)]
    old['scientific_source_sha256'] = api.PARENT_SOURCE
    new['scientific_source_sha256'] = api.CURRENT_SOURCE
    rows = [dict(structure_id=name, energy_hartree=-1.5, converged=True)
            for name in old['structure_ids']]
    heldout_rows = [dict(structure_id=name, energy_hartree=-2.01, converged=True)
                    for name in old['heldout_structure_ids']]
    history = [dict(epoch=i, loss_hartree=-1. + -.5 * i / 41) for i in range(42)]
    metadata = dict(epoch=41, identity=old, loss_hartree=-1.5, molecules=rows,
        history=history, validation_history=[dict(epoch=i, loss_hartree=-2.) for i in (0, 20, 40)])
    class Optimizer:
        def encode(self, state): return json.dumps(state, sort_keys=True).encode()
    optimizer = Optimizer()
    arrays = dict(parameters=np.array([1., 2.]), gradient=np.array([.01, -.02]),
        optimizer=np.frombuffer(optimizer.encode({'count': 41, 'schedule': {'count': 41}}), dtype=np.uint8))
    for name in ('last.npz', 'best.npz'):
        api.write_payload(parent / name, (metadata, arrays))
    validation_metadata = dict(metadata, epoch=40)
    validation_arrays = dict(arrays, parameters=np.array([.9, 1.9]),
        optimizer=np.frombuffer(optimizer.encode({'count': 40, 'schedule': {'count': 40}}), dtype=np.uint8))
    api.write_payload(parent / 'best_validation.npz', (validation_metadata, validation_arrays))
    monkeypatch.setattr(api, 'PARENT_CHECKPOINT', api.file_hash(parent / 'last.npz'))
    proof = {'proof': 'fake source manifest for NumPy transaction control'}
    monkeypatch.setattr(api, 'import_path', lambda *a: SimpleNamespace(verify_sources=lambda *a: proof))
    regression = tmp_path / 'regression.json'
    regression.write_text(json.dumps(dict(status='complete', checkpoint_sha256=api.PARENT_CHECKPOINT,
        original_checkpoint_bytes_unchanged=True, target_repetitions=100, slurm_job_id='94841', backend='gpu',
        provenance=proof, regression_source_sha256=api.file_hash(api.ROOT / 'artifacts/nnao-stage2-20261009-adim6-fix/regress.py'),
        basis_directional_check={'passed': True}, parameter_directional_check={'passed': True},
        runs=[dict(structure_id='ADIM6/AD6', repeat=i, status='passed') for i in range(1, 101)])))
    monkeypatch.setattr(api, 'REGRESSION_HASH', api.file_hash(regression))
    def load(path, identity, optimizer, initial):
        meta, saved = api.read_payload(path)
        assert meta['identity'] == identity
        saved['optimizer'] = json.loads(saved['optimizer'].tobytes())
        return dict(meta, **saved)
    train = SimpleNamespace(load_checkpoint=load, valid_scf=lambda row: row['converged'])
    args = SimpleNamespace(parent_run=parent, parent_manifest=tmp_path / 'parent.json',
                           new_manifest=tmp_path / 'new.json', regression=regression)
    (parent / 'metadata.json').write_text(json.dumps(dict(runtime(), identity=old)))
    (output / 'metadata.json').write_text(json.dumps(dict(runtime(), identity=new)))
    return dict(api=api, train=train, args=args, initial=arrays['parameters'], optimizer=optimizer,
                output_dir=output, identity=new, evaluate=lambda p: (-1.5, arrays['gradient'] + 1e-10, rows),
                heldout=lambda p: (-2.01, heldout_rows)), parent, output, history, arrays


def test_migration_refreshes_gradient_preserving_parent_optimizer_and_history(migration_case):
    kwargs, parent, output, history, old_arrays = migration_case
    api = kwargs.pop('api')
    before = {name: api.file_hash(parent / name) for name in api.CHECKPOINTS}
    api.migrate(**kwargs)
    assert api.output_state(output) == 'resume'
    metadata, arrays = api.read_payload(output / 'last.npz')
    assert metadata['history'] == history and metadata['epoch'] == 41
    assert metadata['validation_history'][-1]['epoch'] == 41
    for name in ('parameters', 'optimizer'):
        np.testing.assert_array_equal(arrays[name], old_arrays[name])
    np.testing.assert_array_equal(arrays['gradient'], old_arrays['gradient'] + 1e-10)
    validation, _ = api.read_payload(output / 'best_validation.npz')
    assert validation['epoch'] == 41
    assert before == {name: api.file_hash(parent / name) for name in api.CHECKPOINTS}


def test_failed_compatibility_check_publishes_no_checkpoint(migration_case):
    kwargs, parent, output, _, arrays = migration_case
    api = kwargs.pop('api')
    kwargs['evaluate'] = lambda p: (-1.5, np.full_like(arrays['gradient'], np.nan), [])
    with pytest.raises(ValueError): api.migrate(**kwargs)
    assert not any((output / name).exists() for name in api.CHECKPOINTS)
    assert json.loads((output / 'migration.json').read_text())['status'] == 'failed'


def test_interrupted_publication_cannot_resume_partial_checkpoint(migration_case, monkeypatch):
    kwargs, parent, output, _, _ = migration_case
    api = kwargs.pop('api'); original = api.os.replace
    def interrupt(source, destination):
        if Path(destination) == output / 'best.npz': raise OSError('injected interrupted publication')
        return original(source, destination)
    monkeypatch.setattr(api.os, 'replace', interrupt)
    with pytest.raises(OSError): api.migrate(**kwargs)
    assert (output / 'last.npz').exists() and not (output / 'best.npz').exists()
    with pytest.raises(ValueError, match='incomplete'): api.output_state(output)
    assert json.loads((output / 'migration.json').read_text())['status'] == 'failed'
