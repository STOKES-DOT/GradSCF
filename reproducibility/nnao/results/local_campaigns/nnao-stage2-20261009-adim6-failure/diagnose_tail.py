"""Replay four original training evaluators, then repeat ADIM6 without updates.

This diagnostic never modifies checkpoints, production sources, solver tolerances,
or the production backward function. Select CUDA/CPU platforms in the shell.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
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

TAIL = ('MCONF/21', 'Amino20x4/LYS_xas', 'BSR36/r16', 'ADIM6/AD6')


def array_hash(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def failure_evidence(error, directory):
    """Read values from the original failing frames; never rerun their graph."""
    evidence = {}; arrays = {}; current = error
    while current is not None:
        trace = current.__traceback__
        while trace is not None:
            frame = trace.tb_frame
            filename = Path(frame.f_code.co_filename).resolve()
            values = frame.f_locals
            if frame.f_code.co_name == 'evaluate':
                if filename == (ROOT / 'tools' / 'train_nnao_pilot.py').resolve():
                    if 'prediction' in values:
                        arrays['original_prediction'] = np.asarray(values['prediction'])
                if filename == (ROOT / 'tools' / 'optimize_methane_nnao.py').resolve():
                    for name in ('energy', 'gradient', 'outputs'):
                        if name in values:
                            arrays['original_' + name] = np.asarray(values[name])
                    if 'info' in values:
                        evidence['original_forward_info'] = {
                            key: finite_number(value) for key, value in values['info'].items()}
                    if 'row' in values:
                        evidence['original_checked_info'] = values['row']
            trace = trace.tb_next
        current = current.__cause__
    if arrays:
        path = directory / 'failure-original-arrays.npz'
        np.savez(path, **arrays)
        evidence['original_arrays_file'] = str(path)
        evidence['original_arrays'] = {
            name: dict(shape=list(value.shape), dtype=str(value.dtype),
                       sha256=array_hash(value), finite=bool(np.isfinite(value).all()),
                       nan_count=int(np.isnan(value).sum()),
                       inf_count=int(np.isinf(value).sum()))
            for name, value in arrays.items()}
    evidence['captured_from_original_traceback'] = bool(arrays)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--candidate-reference', type=Path, required=True,
                        help='candidate.npz from an already verified baseline replay')
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100,
                        help='Total ADIM6/AD6 evaluations after the three-prefix context')
    parser.add_argument('--observed', action='store_true')
    parser.add_argument('--internal-tolerance-factor', type=float, default=1.0,
                        help='Diagnostic only: stricter upstream tolerance; final acceptance is unchanged')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not 0 < args.internal_tolerance_factor <= 1:
        parser.error('--internal-tolerance-factor must be in (0,1]')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'diagnosis.json').exists():
        parser.error('Use a fresh output directory; existing evidence is preserved')
    started = time.perf_counter()
    imports = train._pilot.verify_source_imports(ROOT)
    checkpoint_hash = train._pilot.file_hash(args.checkpoint)
    if checkpoint_hash != args.checkpoint_sha256:
        raise RuntimeError('Checkpoint bytes differ from the frozen accepted checkpoint')
    with np.load(args.checkpoint, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
    identity = metadata['identity']
    if metadata['epoch'] != 41 or identity['implicit_tolerance'] != 1e-8:
        raise RuntimeError('Expected accepted epoch 41 and unchanged 1e-8 tolerance')
    source_hash = train._pilot.scientific_source_hash(ROOT)
    if source_hash != identity['scientific_source_sha256']:
        raise RuntimeError('Scientific source changed since the checkpoint')
    for name, field in (('train_nnao_pilot.py', 'source_sha256'),
                        ('optimize_methane_nnao.py', 'experiment_source_sha256'),
                        ('validate_nnao_pilot.py', 'validation_source_sha256')):
        if train._pilot.file_hash(ROOT / 'tools' / name) != identity[field]:
            raise RuntimeError(f'Training source changed: {name}')
    if train._pilot.file_hash(args.structures) != identity['structures_sha256']:
        raise RuntimeError('Training structure bytes differ from the checkpoint')
    rows = train._pilot.read_structures(args.structures, ids=TAIL)
    if tuple(row['structure_id'] for row in rows) != TAIL:
        raise RuntimeError('Selected structures do not have the original tail order')
    identifiers = identity['structure_ids']
    start = identifiers.index(TAIL[0])
    if tuple(identifiers[start:start + len(TAIL)]) != TAIL:
        raise RuntimeError('Checkpoint training order does not contain this consecutive tail')

    if args.internal_tolerance_factor != 1.0:
        krylov = importlib.import_module('gradscf.solvers.linear.gmres')
        upstream_gmres = krylov.gmres

        def stricter_gmres(*arguments, **controls):
            controls['tol'] *= args.internal_tolerance_factor
            return upstream_gmres(*arguments, **controls)

        krylov.gmres = stricter_gmres

    if args.observed:
        linear = importlib.import_module('gradscf.solvers.linear.implicit')
        original = linear.linear_residual

        def observed(operator, solution, rhs, **controls):
            norm, valid = original(operator, solution, rhs, **controls)
            rhs_norm = jnp.linalg.norm(rhs)
            jax.debug.print(
                'RESPONSE residual={r} rhs_norm={b} candidate_norm={x} '
                'relative_residual={rel} threshold_without_roundoff={t} '
                'rhs_finite={bf} candidate_finite={xf} converged={c} '
                'actual_guard_accepted={ok}', r=norm, b=rhs_norm,
                x=jnp.linalg.norm(solution),
                rel=norm / jnp.maximum(rhs_norm, jnp.finfo(rhs.dtype).tiny),
                t=controls['atol'] + controls['rtol'] * rhs_norm,
                bf=jnp.all(jnp.isfinite(rhs)), xf=jnp.all(jnp.isfinite(solution)),
                c=controls.get('converged', True), ok=valid, ordered=True)
            return norm, valid

        linear.linear_residual = observed

    controls = SimpleNamespace(seed=identity['seed'], cache_dir=args.cache_dir,
        auxbasis=identity['auxbasis'], cache_storage=identity['cache_storage'],
        implicit_tolerance=identity['implicit_tolerance'],
        scf_rescue_level_shift=identity['scf_rescue_level_shift'])
    initial, evaluators, _, manifests, model_config = train.make_evaluators(
        rows, controls, jax, MethaneRHF)
    if json.loads(json.dumps(model_config)) != identity['model_config']:
        raise RuntimeError('Model configuration differs from training')
    for identifier, manifest in manifests.items():
        if manifest != identity['caches'][identifier]:
            raise RuntimeError(f'Cache identity changed: {identifier}')
    optimizer = train.Adam(identity['learning_rate'],
        final_learning_rate=identity['final_learning_rate'],
        epochs=identity['learning_rate_decay_updates'])
    checkpoint = train.load_checkpoint(args.checkpoint, identity, optimizer, initial)
    candidate, _ = optimizer.update(checkpoint['parameters'], checkpoint['gradient'],
                                    checkpoint['optimizer'])
    candidate = np.asarray(candidate)
    with np.load(args.candidate_reference, allow_pickle=False) as reference:
        if not np.array_equal(checkpoint['parameters'], reference['epoch41_parameters']):
            raise RuntimeError('Epoch 41 parameter vector differs from verified baseline')
        if not np.array_equal(candidate, reference['epoch42_parameters']):
            raise RuntimeError('Reconstructed candidate differs from verified baseline')
    if not np.isfinite(candidate).all():
        raise RuntimeError('Candidate parameter vector is not finite')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('This tail-context replay requires an allocated GPU')
    report = dict(status='running', mode='observed' if args.observed else 'baseline',
        internal_tolerance_factor=args.internal_tolerance_factor,
        internal_target=identity['implicit_tolerance'] * args.internal_tolerance_factor,
        final_acceptance_tolerance=identity['implicit_tolerance'],
        diagnostic_source_sha256=train._pilot.file_hash(__file__),
        scope='three original preceding structures followed by repeated original ADIM6 evaluator',
        checkpoint_sha256=checkpoint_hash,
        candidate_reference_sha256=train._pilot.file_hash(args.candidate_reference),
        epoch41_parameters_sha256=array_hash(checkpoint['parameters']),
        epoch41_gradient_sha256=array_hash(checkpoint['gradient']),
        epoch42_parameters_sha256=array_hash(candidate),
        scientific_source_sha256=source_hash, imported_sources=imports,
        python_version=platform.python_version(), jax_version=jax.__version__,
        backend=jax.default_backend(), device_kinds=[d.device_kind for d in jax.devices()],
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        implicit_config=identity['implicit_config'], cache_identities=manifests,
        prefix_ids=list(TAIL[:-1]), target_adim6_calls=args.repeats,
        parameter_update_norm=float(np.linalg.norm(candidate - checkpoint['parameters'])),
        candidate_learning_rate=optimizer.learning_rate_at(checkpoint['epoch']), runs=[])

    # Observe the actual prediction supplied to the original evaluator, only
    # after its original SCF/gradient checks. No additional prediction or JAX
    # arithmetic is executed, and the jitted _value_grad function stays intact.
    predictions = {}; first_predictions = {}
    for identifier, evaluate in evaluators:
        _, experiment = evaluate.__defaults__
        original_evaluate = experiment.evaluate

        def capture_outputs(outputs, original=original_evaluate, identifier=identifier):
            result = original(outputs)
            array = np.asarray(outputs)
            if identifier not in first_predictions:
                first_predictions[identifier] = array.copy()
            predictions[identifier] = dict(
                outputs_sha256=array_hash(array), outputs_finite=bool(np.isfinite(array).all()),
                outputs_shape=list(array.shape), outputs_dtype=str(array.dtype),
                max_output_change_from_first_call=float(np.max(
                    np.abs(array - first_predictions[identifier]))))
            return result

        experiment.evaluate = capture_outputs

    def persist():
        report['elapsed_seconds'] = time.perf_counter() - started
        train._pilot.atomic_json(args.output_dir / 'diagnosis.json', report)

    persist()
    lookup = dict(evaluators)
    sequence = [(identifier, 0) for identifier in TAIL[:-1]] + [
        (TAIL[-1], index) for index in range(1, args.repeats + 1)]
    baseline_gradient = None
    for identifier, repeat in sequence:
        tick = time.perf_counter()
        print(f'BEGIN {identifier} repeat={repeat} mode={report["mode"]}', flush=True)
        try:
            value, gradient, details = lookup[identifier](candidate)
            gradient = np.asarray(gradient)
            if not np.isfinite(value) or not np.isfinite(gradient).all() or not train.valid_scf(details):
                raise RuntimeError(f'Original evaluator returned invalid result: {details}')
            record = dict(structure_id=identifier, repeat=repeat, status='passed',
                energy_hartree=float(value), parameter_gradient_finite=True,
                parameter_gradient_sha256=array_hash(gradient),
                parameter_gradient_norm=float(np.linalg.norm(gradient)),
                forward_details=details, actual_prediction=predictions[identifier],
                elapsed_seconds=time.perf_counter() - tick)
            if identifier == TAIL[-1]:
                if baseline_gradient is None:
                    baseline_gradient = gradient.copy()
                record['max_parameter_gradient_change_from_first_call'] = float(
                    np.max(np.abs(gradient - baseline_gradient)))
        except Exception as error:
            record = dict(structure_id=identifier, repeat=repeat, status='failed',
                error=str(error), traceback=traceback.format_exc(),
                elapsed_seconds=time.perf_counter() - tick,
                **failure_evidence(error, args.output_dir))
            report['runs'].append(record)
            report['status'] = 'failed'
            persist()
            print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
            return 1
        report['runs'].append(record)
        persist()
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    report['status'] = 'complete'
    persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
