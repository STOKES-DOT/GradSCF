"""Explicit one-time source migration followed by the unchanged shared trainer.

The public trainer still verifies the full new 200/50 numerical gates, native
caches and current identity. Migration reads the old run only, re-evaluates all
200 gradients at epoch 41, preserves Adam bytes/counts and old histories, and
publishes into a separate directory. Later invocations use ordinary resume.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARENT_CHECKPOINT = '7e7f66fd900fe50cf9925f6d875fd566a953eb5a67600d67c53c518254084df2'
PARENT_SOURCE = '7f066083658982a552fda47e0144e762796f69788c6a1f1c7e70c77fbfec403a'
CURRENT_SOURCE = '87f14784cbc038349f65f311c4b664809925932abd2f0cccea834c5400102b30'
REGRESSION_HASH = '97f830e904a4cbc9b467e38092da27a712e7957d18ac6cb4b4ac39e11d5a84b8'
IDENTITY_CHANGES = frozenset(('source_sha256', 'experiment_source_sha256',
    'scientific_source_sha256', 'validation_summary_sha256',
    'heldout_validation_summary_sha256'))
CHECKPOINTS = ('last.npz', 'best.npz', 'best_validation.npz')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def output_state(directory):
    directory = Path(directory)
    published = [name for name in CHECKPOINTS if (directory / name).exists()]
    marker = directory / 'migration.json'
    if marker.exists() and json.loads(marker.read_text()).get('status') == 'complete':
        if len(published) != len(CHECKPOINTS):
            raise ValueError('Completed migration has incomplete checkpoint publication.')
        return 'resume'
    if published:
        raise ValueError('Migration is incomplete; inspect partial publication before resuming.')
    return 'fresh'


def verify_identity_transition(old, current):
    changes = {key for key in set(old) | set(current) if old.get(key) != current.get(key)}
    if set(old) != set(current) or changes != IDENTITY_CHANGES:
        raise ValueError(f'Checkpoint identity transition differs from the five allowed fields: {sorted(changes)}')
    return sorted(changes)


def verify_runtime_environment(parent, current):
    packages = ('jax', 'jaxlib', 'flax', 'optax', 'e3nn-jax', 'mace-jax')
    def select(metadata):
        try:
            selected = dict(python_version=metadata['python_version'],
                parameter_dtype=metadata['parameter_dtype'],
                versions={name: metadata['versions'][name] for name in packages})
        except KeyError as error:
            raise ValueError('Runtime environment metadata is incomplete.') from error
        if any(not isinstance(value, str) or not value for value in selected['versions'].values()):
            raise ValueError('Runtime environment package versions are invalid.')
        return selected
    original = select(parent)
    if (original != select(current) or original['python_version'] != '3.11.15'
            or original['parameter_dtype'] != 'float64'
            or any(original['versions'][name] != '0.10.0' for name in ('jax', 'jaxlib'))):
        raise ValueError('Runtime environment differs from the original JAX 0.10.0/Python 3.11.15 checkpoint.')
    return original


def verify_optimizer_counts(state, epoch):
    counts = []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'count': counts.append(int(np.asarray(item)))
                else: visit(item)
        elif hasattr(value, '_fields'):
            for name in value._fields:
                if name == 'count': counts.append(int(np.asarray(getattr(value, name))))
                else: visit(getattr(value, name))
        elif isinstance(value, (tuple, list)):
            for item in value: visit(item)
    visit(state)
    if len(counts) != 2 or any(count != epoch for count in counts):
        raise ValueError(f'Adam and schedule counters must both remain at epoch {epoch}: {counts}')
    return counts


def compatibility_metrics(saved, loss, gradient, records):
    old_records = saved['molecules']; gradient = np.asarray(gradient)
    if [row['structure_id'] for row in records] != [row['structure_id'] for row in old_records]:
        raise ValueError('Re-evaluated batch order differs from the accepted checkpoint.')
    values = np.asarray([row['energy_hartree'] for row in records], dtype=np.float64)
    old_values = np.asarray([row['energy_hartree'] for row in old_records], dtype=np.float64)
    if (not np.isfinite(loss) or not np.isfinite(values).all()
            or gradient.shape != saved['gradient'].shape or gradient.dtype != np.float64
            or not np.isfinite(gradient).all()):
        raise ValueError('Re-evaluated batch contains invalid energy/gradient values.')
    energy_error = float(np.max(np.abs(values - old_values)))
    mean_error = abs(float(loss) - saved['loss_hartree'])
    gradient_passed = bool(np.allclose(gradient, saved['gradient'], atol=2e-6, rtol=2e-5))
    if energy_error > 1e-8 or mean_error > 1e-9 or not gradient_passed:
        raise ValueError(f'Source migration changed the accepted-point energy/gradient: '
                         f'max E={energy_error}, mean E={mean_error}, gradient={gradient_passed}')
    return dict(max_structure_energy_error_hartree=energy_error,
        mean_energy_error_hartree=mean_error, gradient_passed=gradient_passed,
        max_gradient_absolute_error=float(np.max(np.abs(gradient - saved['gradient']))),
        structure_energy_atol_hartree=1e-8, mean_energy_atol_hartree=1e-9,
        gradient_atol=2e-6, gradient_rtol=2e-5)


def verify_regression(report, provenance):
    repeats = [row for row in report.get('runs', []) if row.get('structure_id') == 'ADIM6/AD6']
    if (report.get('status') != 'complete' or str(report.get('slurm_job_id')) != '94841'
            or report.get('backend') != 'gpu' or report.get('checkpoint_sha256') != PARENT_CHECKPOINT
            or report.get('original_checkpoint_bytes_unchanged') is not True
            or report.get('target_repetitions') != 100 or report.get('provenance') != provenance
            or [row.get('repeat') for row in repeats] != list(range(1, 101))
            or any(row.get('status') != 'passed' for row in repeats)
            or any(report.get(key, {}).get('passed') is not True
                   for key in ('basis_directional_check', 'parameter_directional_check'))):
        raise ValueError('The frozen repaired-source GPU regression is incomplete or incompatible.')


def read_payload(path):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {'metadata', 'parameters', 'gradient', 'optimizer'}:
            raise ValueError('Checkpoint payload has unexpected arrays.')
        metadata = json.loads(str(saved['metadata'].item()))
        arrays = {name: np.array(saved[name]) for name in ('parameters', 'gradient', 'optimizer')}
    if (arrays['parameters'].dtype != np.float64 or arrays['gradient'].dtype != np.float64
            or arrays['parameters'].shape != arrays['gradient'].shape
            or not np.isfinite(arrays['parameters']).all() or not np.isfinite(arrays['gradient']).all()
            or arrays['optimizer'].dtype != np.uint8 or arrays['optimizer'].ndim != 1
            or arrays['optimizer'].size == 0):
        raise ValueError('Checkpoint payload dtype/shape/values are invalid.')
    return metadata, arrays


def refreshed_payload(metadata, arrays, identity, lineage, *, loss=None, gradient=None,
                      records=None, validation_history=None):
    metadata = copy.deepcopy(metadata)
    arrays = {key: np.array(value) for key, value in arrays.items()}
    metadata.update(identity=copy.deepcopy(identity), source_transition=copy.deepcopy(lineage))
    if loss is not None: metadata['loss_hartree'] = float(loss)
    if gradient is not None: arrays['gradient'] = np.array(gradient)
    if records is not None: metadata['molecules'] = copy.deepcopy(records)
    if validation_history is not None: metadata['validation_history'] = copy.deepcopy(validation_history)
    return metadata, arrays


def write_payload(path, payload):
    metadata, arrays = payload
    with Path(path).open('wb') as stream:
        np.savez(stream, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_resume_sources(args, marker):
    lineage = marker['source_transition']
    regression = import_path('nnao_resume_source_audit',
        ROOT / 'artifacts/nnao-stage2-20261009-adim6-fix/regress.py')
    provenance = regression.verify_sources(args.parent_manifest, args.new_manifest,
                                            lineage['parent_identity'])
    if provenance != lineage['provenance']:
        raise ValueError('Resumed source/native audit differs from the completed migration provenance.')


def migrate(train, args, initial, evaluate, optimizer, output_dir, identity, heldout):
    directory = Path(output_dir); parent = args.parent_run.resolve()
    if directory.resolve() == parent:
        raise ValueError('Migration must publish into a separate output directory.')
    if output_state(directory) != 'fresh':
        raise ValueError('Migration already completed; use ordinary resume.')
    if identity['scientific_source_sha256'] != CURRENT_SOURCE:
        raise ValueError('Current trainer does not use the frozen repaired scientific source.')
    parent_metadata = parent / 'metadata.json'
    current_metadata = directory / 'metadata.json'
    environment = verify_runtime_environment(json.loads(parent_metadata.read_text()),
                                             json.loads(current_metadata.read_text()))
    hashes = {name: file_hash(parent / name) for name in CHECKPOINTS}
    if hashes['last.npz'] != PARENT_CHECKPOINT:
        raise ValueError('Original epoch-41 checkpoint bytes changed.')
    payloads = {name: read_payload(parent / name) for name in CHECKPOINTS}
    old_identity = payloads['last.npz'][0]['identity']
    if old_identity['scientific_source_sha256'] != PARENT_SOURCE:
        raise ValueError('Parent checkpoint scientific source is not the diagnosed source.')
    changes = verify_identity_transition(old_identity, identity)
    regression_module = import_path('nnao_repair_regression',
        ROOT / 'artifacts/nnao-stage2-20261009-adim6-fix/regress.py')
    provenance = regression_module.verify_sources(args.parent_manifest, args.new_manifest, old_identity)
    if file_hash(args.regression) != REGRESSION_HASH:
        raise ValueError('Frozen job-94841 regression evidence bytes changed.')
    report = json.loads(args.regression.read_text())
    verify_regression(report, provenance)
    if report.get('regression_source_sha256') != file_hash(ROOT / 'artifacts/nnao-stage2-20261009-adim6-fix/regress.py'):
        raise ValueError('Frozen GPU regression script changed.')
    last_metadata, last_arrays = payloads['last.npz']
    for name, (metadata, arrays) in payloads.items():
        if metadata['identity'] != old_identity:
            raise ValueError(f'Parent checkpoint identity differs: {name}')
        checkpoint = train.load_checkpoint(parent / name, old_identity, optimizer, initial)
        verify_optimizer_counts(checkpoint['optimizer'], metadata['epoch'])
        if optimizer.encode(checkpoint['optimizer']) != arrays['optimizer'].tobytes():
            raise ValueError(f'Optimizer serialization changed: {name}')
        if name != 'best_validation.npz' and (metadata['epoch'] != 41 or any(
                not np.array_equal(arrays[key], last_arrays[key]) for key in arrays)):
            raise ValueError('Expected matching last/best epoch-41 accepted states.')
        if name == 'best_validation.npz' and metadata['epoch'] != 40:
            raise ValueError('Expected the original held-out best at epoch 40.')
    if [row['epoch'] for row in last_metadata['history']] != list(range(42)):
        raise ValueError('Parent training history is not the complete epoch 0..41 history.')
    if len(old_identity['structure_ids']) != 200 or len(old_identity['heldout_structure_ids']) != 50 or heldout is None:
        raise ValueError('Migration requires the unchanged complete 200/50 split.')
    lineage = dict(parent_run=str(parent), parent_checkpoint_sha256=hashes,
        parent_identity=old_identity, current_identity=identity, changed_identity_fields=changes,
        runtime_environment=environment, parent_runtime_metadata_sha256=file_hash(parent_metadata),
        provenance=provenance, regression_sha256=file_hash(args.regression), regression_job_id='94841',
        parent_epoch=41, next_epoch=42, history_source_scientific_sha256=PARENT_SOURCE,
        new_history_scientific_sha256=CURRENT_SOURCE,
        note='Epochs 0..41 retain their original measurements. Epoch 41 is re-evaluated without an optimizer update; epoch 42 onward uses the repaired numerical solver.')
    migration = dict(status='running', phase='epoch41_full_batch_remeasurement', source_transition=lineage)
    atomic_json(directory / 'migration.json', migration)
    started = time.perf_counter()
    try:
        parameters = np.array(last_arrays['parameters']); parameters.setflags(write=False)
        loss, gradient, records = evaluate(parameters)
        accepted = dict(last_metadata, **last_arrays)
        metrics = compatibility_metrics(accepted, loss, gradient, records)
        if not np.array_equal(parameters, last_arrays['parameters']):
            raise ValueError('Re-evaluation changed parent parameters.')
        if any(not train.valid_scf(row) for row in records):
            raise ValueError('Re-evaluation contains invalid SCF diagnostics.')
        migration.update(phase='epoch41_heldout_remeasurement', compatibility_metrics=metrics)
        atomic_json(directory / 'migration.json', migration)
        value, heldout_records = heldout(parameters)
        if (not np.isfinite(value) or len(heldout_records) != 50
                or [row['structure_id'] for row in heldout_records] != old_identity['heldout_structure_ids']
                or any(not train.valid_scf(row) for row in heldout_records)):
            raise ValueError('Epoch-41 held-out re-evaluation is invalid.')
        validation_record = dict(epoch=41, loss_hartree=float(value), molecules=heldout_records,
            measurement='source-migration re-evaluation; no optimizer update',
            scientific_source_sha256=CURRENT_SOURCE)
        validation_history = last_metadata['validation_history'] + [validation_record]
        lineage.update(compatibility_metrics=metrics,
            parent_adam_and_schedule_counts=[41, 41], refreshed_gradient=True,
            heldout_epoch41_loss_hartree=float(value))
        refreshed = refreshed_payload(last_metadata, last_arrays, identity, lineage,
            loss=loss, gradient=gradient, records=records, validation_history=validation_history)
        best_validation_metadata, best_validation_arrays = payloads['best_validation.npz']
        best_validation = (refreshed if value < best_validation_metadata['validation_history'][-1]['loss_hartree']
            else refreshed_payload(best_validation_metadata, best_validation_arrays, identity, lineage))
        migration.update(phase='checkpoint_publication', source_transition=lineage,
            heldout_best_epoch=best_validation[0]['epoch'], elapsed_seconds=time.perf_counter() - started)
        atomic_json(directory / 'migration.json', migration)
        staging = Path(tempfile.mkdtemp(prefix='.migration-', dir=directory))
        for name, payload in zip(CHECKPOINTS, (refreshed, refreshed, best_validation)):
            write_payload(staging / name, payload)
            reread = read_payload(staging / name)
            if reread[0] != payload[0] or any(not np.array_equal(reread[1][key], payload[1][key]) for key in payload[1]):
                raise ValueError('Staged checkpoint failed round-trip validation.')
            decoded = train.load_checkpoint(staging / name, identity, optimizer, initial)
            verify_optimizer_counts(decoded['optimizer'], payload[0]['epoch'])
        if hashes != {name: file_hash(parent / name) for name in CHECKPOINTS}:
            raise ValueError('Parent checkpoints changed during migration.')
        for name in CHECKPOINTS:
            if (directory / name).exists():
                raise ValueError('Refusing to overwrite an existing destination checkpoint.')
            os.replace(staging / name, directory / name)
        atomic_json(directory / 'history.json', last_metadata['history'])
        atomic_json(directory / 'validation_history.json', validation_history)
        metadata_path = directory / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['source_transition'] = lineage
        atomic_json(metadata_path, metadata)
        migration.update(status='complete', phase='accepted_epoch41_ready_for_epoch42',
            published_checkpoint_sha256={name: file_hash(directory / name) for name in CHECKPOINTS})
        atomic_json(directory / 'migration.json', migration)
    except Exception as error:
        migration.update(status='failed', error=str(error), traceback=traceback.format_exc(),
                         elapsed_seconds=time.perf_counter() - started)
        atomic_json(directory / 'migration.json', migration)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-run', 'parent-manifest', 'new-manifest', 'regression'):
        parser.add_argument('--' + name, type=Path, required=True)
    args, trainer_argv = parser.parse_known_args(argv)
    if trainer_argv and trainer_argv[0] == '--': trainer_argv = trainer_argv[1:]
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--output-dir', type=Path, required=True)
    output, _ = output_parser.parse_known_args(trainer_argv)
    if output.output_dir.resolve() == args.parent_run.resolve():
        parser.error('Use a separate output directory; the original run is read-only.')
    state = output_state(output.output_dir)
    if '--resume' not in trainer_argv:
        trainer_argv.append('--resume')
    sys.path.insert(0, str(ROOT / 'tools'))
    train = import_path('nnao_shared_training', ROOT / 'tools/train_nnao_pilot.py')
    original = train.run_training
    def run(initial, evaluate, optimizer, output_dir, identity, **kwargs):
        if state == 'fresh':
            migrate(train, args, initial, evaluate, optimizer, output_dir, identity, kwargs.get('heldout'))
        else:
            marker = json.loads((Path(output_dir) / 'migration.json').read_text())
            verify_resume_sources(args, marker)
            if marker['source_transition']['current_identity'] != identity:
                raise ValueError('Current trainer identity differs from the completed source migration.')
            metadata_path = Path(output_dir) / 'metadata.json'
            metadata = json.loads(metadata_path.read_text())
            verify_runtime_environment(marker['source_transition']['runtime_environment'], metadata)
            metadata['source_transition'] = marker['source_transition']
            atomic_json(metadata_path, metadata)
        kwargs['resume'] = True
        return original(initial, evaluate, optimizer, output_dir, identity, **kwargs)
    train.run_training = run
    return train.main(trainer_argv)


if __name__ == '__main__':
    raise SystemExit(main())
