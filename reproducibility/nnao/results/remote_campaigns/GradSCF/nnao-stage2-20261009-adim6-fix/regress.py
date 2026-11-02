"""Regression of the explicit GMRES repair at the frozen epoch-42 candidate.

This reads the old checkpoint without changing its identity or bytes. It uses
the repaired production evaluator and SCF-only fourth-order finite differences.
No solver patching, checkpoint migration, new integrals or training updates occur.
"""
from __future__ import annotations

import argparse
import hashlib
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
jax.config.update('jax_enable_x64', True)
import train_nnao_pilot as train
from optimize_methane_nnao import MethaneRHF

PARENT_SOURCE = '7f066083658982a552fda47e0144e762796f69788c6a1f1c7e70c77fbfec403a'
CHECKPOINT_HASH = '7e7f66fd900fe50cf9925f6d875fd566a953eb5a67600d67c53c518254084df2'
ALLOWED = frozenset(('src/gradscf/solvers/linear/gmres.py',
                     'tools/train_nnao_pilot.py', 'tools/optimize_methane_nnao.py'))
TAIL = ('MCONF/21', 'Amino20x4/LYS_xas', 'BSR36/r16', 'ADIM6/AD6')


def array_hash(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def manifest_scientific_hash(manifest):
    digest = hashlib.sha256()
    prefix = 'src/gradscf/'
    for name, sha in sorted(manifest['files'].items()):
        if name.startswith(prefix) and Path(name).suffix in {'.py', '.json', '.dat'}:
            digest.update(name[len(prefix):].encode())
            digest.update(sha.encode())
    return digest.hexdigest()


def verify_sources(parent_path, repaired_path, identity):
    parent = json.loads(parent_path.read_text())
    repaired = json.loads(repaired_path.read_text())
    if parent['scientific_source_sha256'] != PARENT_SOURCE:
        raise RuntimeError('Parent manifest does not describe the diagnosed production source.')
    if identity['scientific_source_sha256'] != PARENT_SOURCE:
        raise RuntimeError('Checkpoint was not produced by the diagnosed parent source.')
    if set(parent['allowed_changes']) != ALLOWED:
        raise RuntimeError('Parent manifest changes the authorized repair scope.')
    if set(parent['files']) != set(repaired['files']):
        raise RuntimeError('Repair changed source inventory.')
    changed = sorted(name for name in parent['files']
                     if parent['files'][name] != repaired['files'][name])
    if set(changed) != ALLOWED:
        raise RuntimeError(f'Repair changes differ from the explicit three-file scope: {changed}')
    for manifest in (parent, repaired):
        if manifest_scientific_hash(manifest) != manifest['scientific_source_sha256']:
            raise RuntimeError('Manifest scientific hash is internally inconsistent.')
    for name, expected in repaired['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or train._pilot.file_hash(path) != expected:
            raise RuntimeError(f'Live repaired file differs from its frozen manifest: {name}')
    if train._pilot.scientific_source_hash(ROOT) != repaired['scientific_source_sha256']:
        raise RuntimeError('Live scientific source differs from the frozen repair manifest.')
    for name, field in (('train_nnao_pilot.py', 'source_sha256'),
                        ('optimize_methane_nnao.py', 'experiment_source_sha256'),
                        ('validate_nnao_pilot.py', 'validation_source_sha256')):
        if parent['files']['tools/' + name] != identity[field]:
            raise RuntimeError(f'Parent training-tool identity differs: {name}')
    return dict(parent_manifest_sha256=train._pilot.file_hash(parent_path),
                repaired_manifest_sha256=train._pilot.file_hash(repaired_path),
                parent_scientific_source_sha256=PARENT_SOURCE,
                repaired_scientific_source_sha256=repaired['scientific_source_sha256'],
                changed_paths=changed, verified_file_count=len(repaired['files']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'candidate-reference', 'parent-manifest', 'new-manifest',
                 'structures', 'cache-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'regression.json').exists():
        parser.error('Use a fresh output directory; existing evidence is preserved.')
    started = time.perf_counter()
    if train._pilot.file_hash(args.checkpoint) != CHECKPOINT_HASH:
        raise RuntimeError('Original epoch-41 checkpoint bytes changed.')
    with np.load(args.checkpoint, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
    identity = metadata['identity']
    if (metadata['epoch'] != 41 or identity['implicit_tolerance'] != 1e-8
            or identity['implicit_config'] != dict(tolerance=1e-8, max_iter=100, restart=40)
            or identity['dtype'] != 'float64'):
        raise RuntimeError('Expected original epoch 41 and unchanged float64 implicit-SCF configuration.')
    provenance = verify_sources(args.parent_manifest, args.new_manifest, identity)
    imports = train._pilot.verify_source_imports(ROOT)
    if train._pilot.file_hash(args.structures) != identity['structures_sha256']:
        raise RuntimeError('Original training structures changed.')
    rows = train._pilot.read_structures(args.structures, ids=TAIL)
    if tuple(row['structure_id'] for row in rows) != TAIL:
        raise RuntimeError('Tail structures have a different order.')
    index = identity['structure_ids'].index(TAIL[0])
    if tuple(identity['structure_ids'][index:index + 4]) != TAIL:
        raise RuntimeError('Frozen training order does not contain the original tail context.')
    controls = SimpleNamespace(seed=identity['seed'], cache_dir=args.cache_dir,
        auxbasis=identity['auxbasis'], cache_storage=identity['cache_storage'],
        implicit_tolerance=identity['implicit_tolerance'],
        scf_rescue_level_shift=identity['scf_rescue_level_shift'])
    initial, evaluators, values, manifests, config = train.make_evaluators(
        rows, controls, jax, MethaneRHF)
    if json.loads(json.dumps(config)) != identity['model_config']:
        raise RuntimeError('Model configuration changed.')
    for identifier, manifest in manifests.items():
        if manifest != identity['caches'][identifier]:
            raise RuntimeError(f'Primitive DF cache or integral signature changed: {identifier}')
    optimizer = train.Adam(identity['learning_rate'],
        final_learning_rate=identity['final_learning_rate'],
        epochs=identity['learning_rate_decay_updates'])
    # This deliberately uses the checkpoint's original identity after explicitly
    # proving the source transition above; it never writes a replacement identity.
    checkpoint = train.load_checkpoint(args.checkpoint, identity, optimizer, initial)
    candidate, _ = optimizer.update(checkpoint['parameters'], checkpoint['gradient'],
                                    checkpoint['optimizer'])
    candidate = np.asarray(candidate)
    with np.load(args.candidate_reference, allow_pickle=False) as baseline:
        for value, key in ((checkpoint['parameters'], 'epoch41_parameters'),
                           (candidate, 'epoch42_parameters')):
            if array_hash(value) != array_hash(baseline[key]):
                raise RuntimeError(f'Parameter bytes differ from the verified baseline: {key}')
    if not np.isfinite(candidate).all() or jax.default_backend() != 'gpu':
        raise RuntimeError('Regression requires a finite candidate and an allocated GPU.')
    report = dict(status='running', scope='frozen candidate, original three-prefix context, 100 AD6 repetitions and two FD directions',
        provenance=provenance, checkpoint_sha256=CHECKPOINT_HASH,
        candidate_reference_sha256=train._pilot.file_hash(args.candidate_reference),
        candidate_parameters_sha256=array_hash(candidate),
        candidate_learning_rate=optimizer.learning_rate_at(checkpoint['epoch']),
        original_identity=identity, cache_identities=manifests, imported_sources=imports,
        backend=jax.default_backend(), device_kinds=[device.device_kind for device in jax.devices()],
        python_version=platform.python_version(), jax_version=jax.__version__,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), final_acceptance_tolerance=1e-8,
        regression_source_sha256=train._pilot.file_hash(__file__), target_repetitions=args.repeats,
        runs=[])
    captures = {}; experiments = {}; target_outputs = []; basis_gradients = []; parameter_gradients = []
    for identifier, evaluate in evaluators:
        _, experiment = evaluate.__defaults__
        experiments[identifier] = experiment
        original = experiment.evaluate

        def capture(outputs, original=original, identifier=identifier):
            result = original(outputs)
            captures[identifier] = (np.asarray(outputs).copy(), np.asarray(result[1]).copy())
            return result

        experiment.evaluate = capture

    def persist():
        report['elapsed_seconds'] = time.perf_counter() - started
        train._pilot.atomic_json(args.output_dir / 'regression.json', report)

    persist()
    try:
        lookup = dict(evaluators)
        sequence = [(identifier, 0) for identifier in TAIL[:-1]] + [
            (TAIL[-1], repeat) for repeat in range(1, args.repeats + 1)]
        for identifier, repeat in sequence:
            tick = time.perf_counter()
            print(f'BEGIN {identifier} repeat={repeat}', flush=True)
            energy, gradient, details = lookup[identifier](candidate)
            gradient = np.asarray(gradient)
            outputs, output_gradient = captures[identifier]
            if not np.isfinite(energy) or not np.isfinite(gradient).all() or not train.valid_scf(details):
                raise RuntimeError(f'Original production evaluator returned invalid diagnostics: {details}')
            record = dict(structure_id=identifier, repeat=repeat, status='passed',
                energy_hartree=float(energy), scf=details,
                parameter_gradient_norm=float(np.linalg.norm(gradient)),
                parameter_gradient_sha256=array_hash(gradient), basis_gradient_sha256=array_hash(output_gradient),
                outputs_sha256=array_hash(outputs), elapsed_seconds=time.perf_counter() - tick)
            if identifier == TAIL[-1]:
                target_outputs.append(outputs); basis_gradients.append(output_gradient)
                parameter_gradients.append(gradient)
                record['max_parameter_gradient_change_from_first'] = float(np.max(
                    np.abs(gradient - parameter_gradients[0])))
                np.savez(args.output_dir / 'repeated-arrays.npz', parameters=candidate,
                    outputs=np.stack(target_outputs), basis_gradients=np.stack(basis_gradients),
                    parameter_gradients=np.stack(parameter_gradients))
            report['runs'].append(record); persist()
            print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        experiment = experiments[TAIL[-1]]
        mask = np.zeros(target_outputs[0].shape)
        for atom, slot, exponents in zip(experiment.layout.shell_atoms,
                                        experiment.layout.slots, experiment.layout.parameters.exponents):
            if slot >= 0:
                mask[atom, slot, :len(exponents)] = 1.
        basis_direction = np.random.default_rng(identity['seed']).normal(size=mask.shape) * mask
        basis_direction /= np.linalg.norm(basis_direction)

        def checked_parameter_value(vector):
            value, details = dict(values)[TAIL[-1]](vector)
            if not np.isfinite(value) or not train.valid_scf(details):
                raise RuntimeError(f'Perturbed parameter SCF did not reconverge: {details}')
            return value, details

        norm = np.linalg.norm(parameter_gradients[0])
        if norm <= 1e-12:
            raise RuntimeError('Parameter gradient is too small for its normalized FD direction.')
        parameter_direction = parameter_gradients[0] / norm
        for name, evaluate, point, gradient, direction in (
            ('basis', experiment.evaluate_value, target_outputs[0], basis_gradients[0], basis_direction),
            ('parameter', checked_parameter_value, candidate, parameter_gradients[0], parameter_direction)):
            print(f'BEGIN fourth_order_fd={name}', flush=True)
            result = train._pilot.directional_check(evaluate, point, gradient, direction,
                step=1e-4, atol=2e-6, rtol=2e-5)
            report[name + '_directional_check'] = result
            np.savez(args.output_dir / (name + '-fd.npz'), point=point, gradient=gradient, direction=direction)
            persist()
            if not result['passed']:
                raise RuntimeError(f'{name} implicit gradient failed SCF-only fourth-order finite difference.')
        if train._pilot.file_hash(args.checkpoint) != CHECKPOINT_HASH:
            raise RuntimeError('Regression changed the original checkpoint bytes.')
        report['original_checkpoint_bytes_unchanged'] = True
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        persist()
        raise
    persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
