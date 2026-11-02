"""Audited one-time native-library migration; retain the strict resume loader."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace

import jax
import numpy as np
import train_nnao_joint_ad as training

trainer = training.base.trainer
pilot = training.base.pilot


def migrate_checkpoint(source, target, old_identity, new_identity, optimizer, initial):
    checkpoint = trainer.load_checkpoint(source, old_identity, optimizer, initial)
    epoch = checkpoint['epoch']
    rows = checkpoint['molecules']
    if ([r['structure_id'] for r in rows] != old_identity['structure_ids']
            or not all(trainer.valid_scf(r) for r in rows)
            or not np.isfinite([r['energy_hartree'] for r in rows]).all()
            or not np.isclose(np.mean([r['energy_hartree'] for r in rows]),
                              checkpoint['loss_hartree'], atol=1e-10, rtol=0)):
        raise ValueError('Checkpoint is not a complete accepted batch')
    if [r['epoch'] for r in checkpoint['history']] != list(range(epoch+1)):
        raise ValueError('Checkpoint history does not match accepted epoch')
    counts = [int(x) for x in jax.tree.leaves(checkpoint['optimizer'])
              if np.shape(x) == () and np.issubdtype(np.asarray(x).dtype, np.integer)]
    if not counts or any(count != epoch for count in counts):
        raise ValueError('Adam step count does not match accepted epoch')
    if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(checkpoint['optimizer'])):
        raise ValueError('Nonfinite Adam state')
    if np.any(np.abs(checkpoint['parameters'][optimizer.ntheta:]) > old_identity['configuration']['log_bound']+1e-12):
        raise ValueError('Invalid exponent scales')
    if Path(target).exists():
        raise FileExistsError(target)
    checkpoint['identity'] = new_identity
    trainer.save_checkpoint(target, checkpoint, optimizer)
    # Serialization must preserve moments, not just reconstruct a valid state.
    with np.load(source, allow_pickle=False) as a, np.load(target, allow_pickle=False) as b:
        for key in ('parameters', 'gradient', 'optimizer'):
            np.testing.assert_array_equal(a[key], b[key])
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', type=Path, default=Path(__file__).resolve().parent)
    args0 = parser.parse_args()
    task = args0.task_dir
    deployment = json.loads((task/'deployment.json').read_text())
    root = Path(deployment['source'])
    assert training.ROOT.resolve() == root.resolve()
    for name, digest in deployment['files'].items():
        assert pilot.file_hash(root/name) == digest, name
    for name, digest in deployment['archive_files'].items():
        assert pilot.file_hash(task/'archive'/name) == digest, name
    source = task/'archive/last.npz'
    with np.load(source, allow_pickle=False) as saved:
        parent = json.loads(str(saved['metadata'].item()))
    old = parent['identity']
    assert old['native_library_sha256'] == deployment['old_native_sha256']
    assert old['model_config'] == training.MODEL_CONFIG
    assert old['exponent_keys'] == [list(k) for k in training.base.EXPONENT_KEYS]
    config = dict(old['configuration'], output_dir=str(task/'training'),
                  compilation_cache_dir=str(task/'compilation-cache'))
    args = SimpleNamespace(**config)
    for key in ('structures', 'checkpoint', 'output_dir', 'compilation_cache_dir'):
        setattr(args, key, Path(getattr(args, key)))
    assert pilot.file_hash(args.structures) == old['structures_sha256']
    assert pilot.file_hash(args.checkpoint) == old['parent_checkpoint_sha256']
    rows = pilot.read_structures(args.structures)
    assert [r['structure_id'] for r in rows] == old['structure_ids'] and len(rows) == 200
    jax.config.update('jax_enable_x64', True)
    theta, beta, _ = training.read_initial_parameters(args.checkpoint, log_bound=args.log_bound)
    initial = np.concatenate((theta, beta))
    mask = np.ones(len(theta), dtype=bool)
    mask[old['frozen_parameter_indices']] = False
    optimizer = training.JointAdam(len(theta), learning_rate=args.learning_rate,
        exponent_learning_rate=args.exponent_learning_rate, log_bound=args.log_bound,
        trainable_mask=mask)
    native = root/'src/gradscf/integrals/_native/lib/libgradscf_integrals.so'
    assert pilot.file_hash(native) == deployment['native_sha256']
    from gradscf.tools.jax_runtime import configure_jax_persistent_cache
    cache = configure_jax_persistent_cache(cache_dir=str(training.compilation_cache_path(args, deployment['native_sha256'])))
    identity = deepcopy(old)
    identity.update(configuration=config, optimizer_reset=False,
        scientific_source_sha256=pilot.scientific_source_hash(root),
        driver_sha256=pilot.file_hash(root/'tools/train_nnao_joint_ad.py'),
        native_library_sha256=deployment['native_sha256'],
        compilation_cache_dir=cache, imported_sources=pilot.verify_source_imports(root),
        migration=dict(source_checkpoint_sha256=pilot.file_hash(source),
            accepted_epoch=parent['epoch'], old_identity=old,
            launcher_sha256=pilot.file_hash(__file__), native_threads=os.environ['GRADSCF_NATIVE_THREADS']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not (args.output_dir/'last.npz').exists():
        # last.npz is committed after best.npz, so a partial migration cannot resume.
        migrate_checkpoint(task/'archive/best.npz', args.output_dir/'best.npz', old, identity, optimizer, initial)
        migrate_checkpoint(source, args.output_dir/'last.npz', old, identity, optimizer, initial)
    accepted = trainer.load_checkpoint(args.output_dir/'last.npz', identity, optimizer, initial)
    pilot.atomic_json(args.output_dir/'metadata.json', identity)
    pilot.atomic_json(args.output_dir/'runtime.json', dict(host=platform.node(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), jax_version=jax.__version__,
        devices=[str(d) for d in jax.devices()], device_kinds=[d.device_kind for d in jax.devices()],
        native_threads=os.environ['GRADSCF_NATIVE_THREADS'], native_sha256=deployment['native_sha256']))
    print('MIGRATION_VERIFIED', json.dumps(dict(accepted_epoch=accepted['epoch'],
        loss_hartree=accepted['loss_hartree'], optimizer_preserved=True)), flush=True)
    engine = training.NativeJointEngine(rows, args, old, None)
    np.testing.assert_array_equal(engine.trainable_mask, mask)
    reference = {r['structure_id']: r['energy_hartree'] for r in accepted['molecules']}
    gates = []
    for key in ('W4-11/n2', 'S22/19a'):
        value, gradient, info = engine.molecule(accepted['parameters'], key)
        assert trainer.valid_scf(info) and np.isfinite(gradient).all(), key
        assert abs(value-reference[key]) <= 1e-7, (key, value, reference[key])
        gates.append(dict(structure_id=key, energy_hartree=value,
            reference_hartree=reference[key], absolute_error_hartree=abs(value-reference[key]), diagnostics=info))
        pilot.atomic_json(task/'restart-gates.json', gates)
        print('RESTART_GATE', json.dumps(gates[-1]), flush=True)
    engine.batch = accepted['epoch']+1
    print('RESUME_TRAINING', accepted['epoch']+1, 'through', args.epochs, flush=True)
    summary = trainer.run_training(initial, engine.evaluate, optimizer, args.output_dir,
        identity, epochs=args.epochs, resume=True)
    if summary['status'] != 'complete':
        raise RuntimeError(summary)


if __name__ == '__main__':
    main()
