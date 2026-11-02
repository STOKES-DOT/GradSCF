"""Replay the failed Adam candidate without changing the training checkpoint."""
import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
os.environ['JAX_ENABLE_X64'] = '1'
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import train_nnao_pilot as train
from optimize_methane_nnao import MethaneRHF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--observed', action='store_true')
    parser.add_argument('--candidate-repeats', type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    imports = train._pilot.verify_source_imports(ROOT)
    with np.load(args.checkpoint, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
    identity = metadata['identity']
    source_hash = train._pilot.scientific_source_hash(ROOT)
    if source_hash != identity['scientific_source_sha256']:
        raise RuntimeError('Scientific source changed since the accepted checkpoint.')
    for name, expected in [('train_nnao_pilot.py', identity['source_sha256']),
                           ('optimize_methane_nnao.py', identity['experiment_source_sha256'])]:
        if train._pilot.file_hash(ROOT / 'tools' / name) != expected:
            raise RuntimeError(f'Training source changed: {name}')
    if metadata['epoch'] != 41 or identity['implicit_tolerance'] != 1e-8:
        raise RuntimeError('Expected epoch41 checkpoint and original 1e-8 response tolerance.')

    if args.observed:
        linear = importlib.import_module('gradscf.solvers.linear.implicit')
        original = linear.linear_residual

        def observed(operator, solution, rhs, **controls):
            residual, valid = original(operator, solution, rhs, **controls)
            rhs_norm = jnp.linalg.norm(rhs)
            threshold = controls['atol'] + controls['rtol'] * rhs_norm
            # Read the actual production guard. No additional operator call.
            # Its threshold also includes the production roundoff allowance.
            jax.debug.print('RESPONSE residual={r} rhs_norm={b} threshold_without_roundoff={t} '
                            'relative_residual={rel} finite_solution={f} '
                            'converged={c} actual_guard_accepted={ok}',
                            r=residual, b=rhs_norm, t=threshold,
                            rel=residual / jnp.maximum(rhs_norm, jnp.finfo(rhs.dtype).tiny),
                            f=jnp.all(jnp.isfinite(solution)),
                            c=controls.get('converged', True), ok=valid, ordered=True)
            return residual, valid

        linear.linear_residual = observed

    controls = SimpleNamespace(seed=identity['seed'], cache_dir=args.cache_dir,
        auxbasis=identity['auxbasis'], cache_storage=identity['cache_storage'],
        implicit_tolerance=identity['implicit_tolerance'],
        scf_rescue_level_shift=identity['scf_rescue_level_shift'])
    rows = train._pilot.read_structures(args.structures, ids=['ADIM6/AD6'])
    initial, evaluators, _, manifests, config = train.make_evaluators(
        rows, controls, jax, MethaneRHF)
    if json.loads(json.dumps(config)) != identity['model_config']:
        raise RuntimeError('MACE model configuration changed.')
    if manifests['ADIM6/AD6'] != identity['caches']['ADIM6/AD6']:
        raise RuntimeError('ADIM6 cache identity changed.')
    optimizer = train.Adam(identity['learning_rate'],
        final_learning_rate=identity['final_learning_rate'],
        epochs=identity['learning_rate_decay_updates'])
    checkpoint = train.load_checkpoint(args.checkpoint, identity, optimizer, initial)
    candidate, _ = optimizer.update(checkpoint['parameters'], checkpoint['gradient'],
                                    checkpoint['optimizer'])
    candidate = np.asarray(candidate)
    train_evaluate = evaluators[0][1]
    graph, experiment = train_evaluate.__defaults__
    closure = inspect.getclosurevars(train_evaluate).nonlocals
    predict, backward = closure['predict'], closure['backward']
    report = dict(mode='observed' if args.observed else 'baseline',
        checkpoint_sha256=train._pilot.file_hash(args.checkpoint),
        scientific_source_sha256=source_hash, imported_sources=imports,
        python_version=platform.python_version(), jax_version=jax.__version__,
        backend=jax.default_backend(), device_kinds=[d.device_kind for d in jax.devices()],
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        parameter_count=int(initial.size), implicit_config=identity['implicit_config'],
        candidate_learning_rate=optimizer.learning_rate_at(checkpoint['epoch']),
        parameter_update_norm=float(np.linalg.norm(candidate-checkpoint['parameters'])),
        cache_identity=manifests['ADIM6/AD6'], runs=[])
    np.savez(args.output_dir / 'candidate.npz', epoch41_parameters=checkpoint['parameters'],
             epoch42_parameters=candidate)
    train._pilot.atomic_json(args.output_dir / 'diagnosis.json', report)
    runs = [('epoch41', checkpoint['parameters'])] + [
        (f'epoch42_repeat_{i}', candidate) for i in range(args.candidate_repeats)]
    for label, vector in runs:
        print(f'BEGIN {label} mode={report["mode"]}', flush=True)
        tick = time.perf_counter()
        outputs = predict(jnp.asarray(vector), graph)
        (energy, info), gradient = experiment._value_grad(
            outputs, experiment.ps, experiment.ph, experiment.rep)
        gradient = np.asarray(gradient)
        row = dict(label=label, energy_hartree=float(energy),
            forward_details=experiment._checked_info(energy, info),
            outputs_finite=bool(np.isfinite(np.asarray(outputs)).all()),
            basis_gradient_finite=bool(np.isfinite(gradient).all()),
            nan_count=int(np.isnan(gradient).sum()), inf_count=int(np.isinf(gradient).sum()),
            gradient_size=int(gradient.size), elapsed_seconds=time.perf_counter()-tick)
        if row['basis_gradient_finite']:
            parameter_gradient = np.asarray(backward(jnp.asarray(vector), graph, jnp.asarray(gradient)))
            row.update(basis_gradient_norm=float(np.linalg.norm(gradient)),
                parameter_gradient_finite=bool(np.isfinite(parameter_gradient).all()),
                parameter_gradient_norm=float(np.linalg.norm(parameter_gradient)))
        report['runs'].append(row)
        report['elapsed_seconds'] = time.perf_counter()-started
        train._pilot.atomic_json(args.output_dir / 'diagnosis.json', report)
        print(json.dumps(row, sort_keys=True), flush=True)
    report['status'] = 'complete'
    train._pilot.atomic_json(args.output_dir / 'diagnosis.json', report)


if __name__ == '__main__':
    main()
