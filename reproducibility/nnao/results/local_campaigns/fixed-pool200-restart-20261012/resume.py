"""Checked source migration of the accepted stage-C state; no optimizer reset."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT_SOURCE = '06d826104d2e472ef4d26bdfafb38a9888c1ce980d0f7d42f515f4d15999a290'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def read_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'].item()))
        arrays = {name: np.array(data[name]) for name in ('parameters', 'gradient', 'optimizer')}
    return metadata, arrays


def migrate_checkpoint(source, target, identity, *, parent_source=PARENT_SOURCE,
                       allow_template_change=False):
    metadata, arrays = read_checkpoint(source)
    parent = metadata['identity']
    if parent['scientific_source_sha256'] != parent_source:
        raise ValueError('Unexpected parent source identity.')
    changed = {key for key in set(parent)|set(identity) if parent.get(key) != identity.get(key)}
    allowed = {'scientific_source_sha256'}
    if allow_template_change:
        allowed.add('initialization_sha256')
    if 'scientific_source_sha256' not in changed or not changed <= allowed:
        raise ValueError(f'Migration may change only source and an explicitly checked template, got {changed}.')
    if arrays['parameters'].shape != arrays['gradient'].shape:
        raise ValueError('Invalid parameter/gradient shape.')
    if not all(np.isfinite(arrays[key]).all() for key in ('parameters', 'gradient')):
        raise ValueError('Nonfinite accepted checkpoint.')
    target = Path(target)
    if target.exists():
        raise ValueError('Do not overwrite an existing migrated checkpoint.')
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata['identity'] = identity
    metadata['source_migration'] = dict(parent_sha256=sha(source), parent_source=parent_source,
        accepted_epoch=metadata['epoch'], optimizer_reset=False)
    temporary = target.with_name(target.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    os.replace(temporary, target)
    restored_metadata, restored = read_checkpoint(target)
    for name in arrays:
        assert arrays[name].dtype == restored[name].dtype
        assert arrays[name].shape == restored[name].shape
        assert arrays[name].tobytes() == restored[name].tobytes()
    assert restored_metadata['history'] == metadata['history']
    return dict(epoch=metadata['epoch'], loss_hartree=metadata['loss_hartree'],
        parent_sha256=sha(source), migrated_sha256=sha(target),
        parameters_preserved=True, gradient_preserved=True, optimizer_bytes_preserved=True,
        optimizer_reset=False, history_preserved=True,
        initializer_fingerprint_changed='initialization_sha256' in changed,
        parent_initializer_sha256=parent.get('initialization_sha256'),
        template_initializer_sha256=identity.get('initialization_sha256'))


def verify_replay(parent, arrays, result):
    value, gradient, rows = result
    expected = {row['structure_id']: row['energy_hartree'] for row in parent['molecules']}
    if len(rows) != len(expected) or {r['structure_id'] for r in rows} != set(expected):
        raise ValueError('Replay molecule set differs from accepted checkpoint.')
    error = max(abs(row['energy_hartree']-expected[row['structure_id']]) for row in rows)
    if error > 1e-8:
        raise ValueError(f'Per-molecule replay energy changed by {error} Hartree.')
    np.testing.assert_allclose(value, parent['loss_hartree'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(gradient, arrays['gradient'], atol=1e-7, rtol=1e-5)
    return dict(passed=True, molecule_count=len(rows), parent_epoch=parent['epoch'],
        max_molecular_energy_error_hartree=error,
        mean_energy_error_hartree=abs(value-parent['loss_hartree']),
        max_gradient_error=float(np.max(np.abs(gradient-arrays['gradient']))))


def prepare():
    from gradscf.model.nnao.runtime import scientific_source_hash
    from gradscf.model.nnao.training.trainer import make_mace_predictor
    validation = ROOT/'artifacts/fixed-pool-performance-20261012/full200'
    summary = json.loads((validation/'summary.json').read_text())
    gate = json.loads((validation/'gate.json').read_text())
    if summary['status'] != 'complete' or summary['accepted_epoch'] != 1:
        raise ValueError('Complete optimized training validation is required.')
    if not gate['passed'] or len(gate['parameter_direction_checks']) != 200:
        raise ValueError('Complete 200-molecule gradient gate is required.')
    identity = dict(summary['identity'])
    if scientific_source_hash(ROOT) != identity['scientific_source_sha256']:
        raise ValueError('Production source differs from the validated optimized source.')
    # Initial values are only an unravel/serialization template on resume.
    # Reconstruct on the actual allocated device, never replace trained arrays.
    initial, _, _ = make_mace_predictor(identity['seed'], identity['model_config'])
    parent, arrays = read_checkpoint(OUT/'archive/last.npz')
    initial = np.asarray(initial)
    if initial.shape != arrays['parameters'].shape or initial.dtype != arrays['parameters'].dtype:
        raise ValueError('Runtime parameter template shape/dtype changed.')
    identity['initialization_sha256'] = hashlib.sha256(initial.tobytes()).hexdigest()
    save(OUT/'template-check.json', dict(parent_sha256=parent['identity']['initialization_sha256'],
        validated_sha256=summary['identity']['initialization_sha256'],
        runtime_sha256=identity['initialization_sha256'], shape=list(initial.shape),
        dtype=str(initial.dtype), template_values_not_used_for_training=True))
    records = {name: migrate_checkpoint(OUT/'archive'/name, OUT/'training'/name, identity,
                                        allow_template_change=True)
               for name in ('last.npz', 'best.npz')}
    save(OUT/'migration.json', dict(status='complete', checkpoints=records,
        validation_summary_sha256=sha(validation/'summary.json'),
        target_source=identity['scientific_source_sha256'], target_epochs=1000))
    print(json.dumps(records), flush=True)


def run():
    import jax
    from gradscf.model.nnao.training import trainer
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    migration = json.loads((OUT/'migration.json').read_text())
    if migration['status'] != 'complete':
        raise ValueError('Migration has not completed.')
    parent, arrays = read_checkpoint(OUT/'archive/last.npz')
    accepted, current = read_checkpoint(OUT/'training/last.npz')
    if accepted['epoch'] != parent['epoch']:
        raise ValueError('This migration launcher only starts from the archived accepted epoch.')
    for key in arrays:
        assert arrays[key].tobytes() == current[key].tobytes()
    optimizer = trainer.Adam(accepted['identity']['learning_rate'])
    state = optimizer.decode(current['optimizer'].tobytes(), optimizer.init(current['parameters']))
    counts = [int(x) for x in jax.tree.leaves(state)
              if np.ndim(x) == 0 and np.asarray(x).dtype.kind in 'iu']
    if not counts or any(count != parent['epoch'] for count in counts):
        raise ValueError(f'Adam counters do not match accepted epoch: {counts}')
    save(OUT/'optimizer-check.json', dict(epoch=parent['epoch'], counts=counts,
        bytes_preserved=True, parameters_preserved=True, gradient_preserved=True))
    # This task-local guard only observes the normal trainer callback. It does
    # not replace SCF, gradients, parameters, optimizer state or acceptance.
    original_batch = trainer.evaluate_batch
    replays = []
    def guarded_batch(vector, evaluators, **kwargs):
        result = original_batch(vector, evaluators, **kwargs)
        if np.array_equal(vector, arrays['parameters']):
            replays.append(verify_replay(parent, arrays, result))
            save(OUT/'restart-gates.json', dict(passed=True, replays=replays))
        return result
    trainer.evaluate_batch = guarded_batch
    identity = accepted['identity']
    result = run_fixed_pool(identity['references'], identity['candidate'],
        ROOT/'artifacts/fixed-pool200-20261012/cache', OUT/'training', epochs=1000,
        learning_rate=identity['learning_rate'], seed=identity['seed'],
        storage=identity['storage'], resume=True)
    return int(result['status'] != 'complete')


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'start':
            prepare()
            raise SystemExit(run())
        elif sys.argv[1] == 'prepare':
            prepare()
        elif sys.argv[1] == 'run':
            raise SystemExit(run())
        else:
            raise ValueError('Use prepare or run.')
    except Exception as error:
        save(OUT/'restart-failure.json', dict(error=str(error)))
        raise
