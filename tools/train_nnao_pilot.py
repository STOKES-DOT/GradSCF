"""Train one shared MACE contraction model on the validated GMTKN55 RHF pilot.

The loss is the equal-weight mean of reconverged total energies. Geometry,
primitive exponents and core6 remain fixed. Native DF
caches are prepared separately; this process requires CUDA and stores no ERI.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import time
import traceback

import numpy as np

_spec = importlib.util.spec_from_file_location(
    'nnao_pilot_validation', Path(__file__).with_name('validate_nnao_pilot.py'))
_pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pilot)


def valid_scf(details):
    keys = ('orbital_residual', 'min_overlap_eigenvalue', 'reconstruction_error', 'fixed_point_residual')
    return (details.get('converged') is True and all(key in details for key in keys)
        and np.isfinite([details[key] for key in keys]).all()
        and details['orbital_residual'] <= 1e-7
        and details['fixed_point_residual'] <= 1e-7
        and details['min_overlap_eigenvalue'] >= 1e-8
        and details['reconstruction_error'] <= 1e-9)


def verify_validation(path, structures, identifiers, *, tolerance, rescue_level_shift=0.0):
    """Require a complete successful gate for exactly the training structures."""
    summary = json.loads(Path(path).read_text())
    results = summary.get('results', [])
    if (summary.get('config', {}).get('structures_sha256') != _pilot.file_hash(structures)
            or sorted(row['structure_id'] for row in results) != sorted(identifiers)):
        raise ValueError('Validation structures differ from the training input.')
    if (summary.get('exit_code') != 0 or summary.get('failed') != 0
            or summary.get('passed') != len(identifiers)
            or any(row.get('status') != 'passed' for row in results)):
        raise ValueError('A complete successful validation is required before training.')
    if (summary.get('config', {}).get('implicit_tolerance') != tolerance
            or any(row.get('implicit_config', {}).get('tolerance') != tolerance for row in results)):
        raise ValueError('Validation implicit tolerance differs from the training configuration.')
    if (summary.get('config', {}).get('scf_rescue_level_shift', 0.) != rescue_level_shift
            or any(row.get('scf_rescue_level_shift', 0.) != rescue_level_shift for row in results)):
        raise ValueError('Validation SCF rescue policy differs from the training configuration.')
    if any(row.get(key, {}).get('passed') is not True for row in results
           for key in ('coefficient_directional_check', 'parameter_directional_check')):
        raise ValueError('Validation must include passing coefficient and parameter gradient checks.')
    if any(row.get('implicit_gradient_finite') is not True for row in results):
        raise ValueError('Validation requires finite implicit gradients for every molecule.')
    for row in results:
        history = row.get('adam_history', [])
        if row.get('accepted_adam_steps') != 5 or [item.get('step') for item in history] != list(range(6)):
            raise ValueError('Validation requires five accepted Adam steps for every molecule.')
        if not valid_scf(row.get('initial_scf', {})) or any(not valid_scf(item) for item in history):
            raise ValueError('Validation contains missing or invalid SCF diagnostics.')
    return summary


def verify_validation_sources(summary, root, *, scientific_hash):
    if summary['config'].get('scientific_source_sha256') != scientific_hash:
        raise ValueError('Scientific sources changed since validation; rerun the gate.')
    for field, filename in (('experiment_source_sha256', 'optimize_methane_nnao.py'),
                            ('source_sha256', 'validate_nnao_pilot.py')):
        if summary['config'].get(field) != _pilot.file_hash(Path(root) / 'tools' / filename):
            raise ValueError(f'{filename} changed since validation; rerun the gate.')


def evaluate_batch(vector, evaluators, *, progress=None):
    """Evaluate every molecule at the same vector; never skip invalid samples."""
    records = []; gradient_sum = np.zeros_like(np.asarray(vector)); energies = []
    if not evaluators:
        raise ValueError('The training batch is empty.')
    for identifier, evaluate in evaluators:
        try:
            value, gradient, details = evaluate(vector)
            gradient = np.asarray(gradient)
            valid = (np.isfinite(value) and gradient.shape == gradient_sum.shape
                and np.isfinite(gradient).all() and valid_scf(details))
            if not valid:
                raise RuntimeError(f'Invalid SCF/gradient diagnostics: {details}')
        except Exception as error:
            failure = RuntimeError(f'{identifier}: {error}')
            failure.structure_id = identifier
            raise failure from error
        energies.append(float(value)); gradient_sum += gradient
        records.append(dict(structure_id=identifier, energy_hartree=float(value),
                            parameter_gradient_norm=float(np.linalg.norm(gradient)), **details))
        if progress is not None:
            progress(records)
    return float(np.mean(energies)), gradient_sum / len(evaluators), records


def evaluate_values(vector, evaluators, *, progress=None):
    """Held-out diagnostics use the same SCF primal, without a gradient or update."""
    if not evaluators:
        raise ValueError('The held-out batch is empty.')
    records = []
    for identifier, evaluate in evaluators:
        try:
            value, details = evaluate(vector)
            if not np.isfinite(value) or not valid_scf(details):
                raise RuntimeError(f'Invalid held-out SCF diagnostics: {details}')
        except Exception as error:
            failure = RuntimeError(f'{identifier}: {error}')
            failure.structure_id = identifier
            raise failure from error
        records.append(dict(structure_id=identifier, energy_hartree=float(value), **details))
        if progress is not None:
            progress(records)
    return float(np.mean([row['energy_hartree'] for row in records])), records


def save_checkpoint(path, checkpoint, optimizer):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in checkpoint.items()
                if key not in {'parameters', 'gradient', 'optimizer'}}
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, parameters=np.asarray(checkpoint['parameters']),
            gradient=np.asarray(checkpoint['gradient']),
            optimizer=np.frombuffer(optimizer.encode(checkpoint['optimizer']), dtype=np.uint8),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    os.replace(temporary, path)


def load_checkpoint(path, identity, optimizer, initial):
    with np.load(path, allow_pickle=False) as saved:
        checkpoint = json.loads(str(saved['metadata'].item()))
        if checkpoint.get('identity') != identity:
            raise ValueError('Checkpoint identity differs: data, source, cache or configuration changed.')
        checkpoint.update(parameters=np.array(saved['parameters']), gradient=np.array(saved['gradient']),
                          optimizer=optimizer.decode(saved['optimizer'].tobytes(), optimizer.init(initial)))
    if (checkpoint['parameters'].shape != np.asarray(initial).shape
            or checkpoint['gradient'].shape != np.asarray(initial).shape
            or not np.isfinite(checkpoint['parameters']).all()
            or not np.isfinite(checkpoint['gradient']).all()):
        raise ValueError('Checkpoint parameter/gradient shape or values are invalid.')
    return checkpoint


def save_failure_snapshot(path, vector, error, summary):
    """Save the unaccepted trial, following explicit diagnostics through wrappers."""
    arrays = dict(parameters=np.asarray(vector)); metadata = dict(summary)
    cause = error
    while cause is not None:
        if hasattr(cause, 'structure_id'):
            metadata.setdefault('structure_id', cause.structure_id)
        if hasattr(cause, 'failure_arrays'):
            arrays.update(cause.failure_arrays)
            metadata['details'] = cause.failure_details
        cause = cause.__cause__
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays,
                 metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    os.replace(temporary, path)


def run_training(initial, evaluate, optimizer, output_dir, identity, *, epochs, resume=False,
                 heldout=None, validation_interval=20):
    """One complete batch and one Adam update per epoch, with accepted checkpoints."""
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    last = directory / 'last.npz'; best = directory / 'best.npz'
    validation_best = directory / 'best_validation.npz'
    if validation_interval < 1:
        raise ValueError('validation_interval must be positive.')
    if last.exists() and not resume:
        raise ValueError('Output already contains a checkpoint; use --resume or a new output directory.')
    if resume and not last.exists():
        raise ValueError('Resume requires an existing last.npz checkpoint.')
    checkpoint = None; vector = None; started = time.perf_counter(); attempted_epoch = 0
    try:
        if resume:
            checkpoint = load_checkpoint(last, identity, optimizer, initial)
            vector = checkpoint['parameters']
        else:
            vector = initial
            loss, gradient, records = evaluate(vector)
            candidate = dict(identity=identity, parameters=np.asarray(initial),
                optimizer=optimizer.init(initial), gradient=np.asarray(gradient), epoch=0,
                loss_hartree=loss, molecules=records, history=[], validation_history=[])
            candidate['history'].append(dict(epoch=0, loss_hartree=loss, molecules=records))
            if heldout is not None:
                value, heldout_records = heldout(initial)
                candidate['validation_history'].append(dict(epoch=0, loss_hartree=value, molecules=heldout_records))
                save_checkpoint(validation_best, candidate, optimizer)
            checkpoint = candidate
            save_checkpoint(last, checkpoint, optimizer); save_checkpoint(best, checkpoint, optimizer)
        checkpoint.setdefault('validation_history', [])
        best_loss = load_checkpoint(best, identity, optimizer, initial)['loss_hartree']
        best_validation_loss = (load_checkpoint(validation_best, identity, optimizer, initial)['validation_history'][-1]['loss_hartree']
                                if heldout is not None else None)
        _pilot.atomic_json(directory / 'history.json', checkpoint['history'])
        if heldout is not None:
            _pilot.atomic_json(directory / 'validation_history.json', checkpoint['validation_history'])
        initial_loss = checkpoint['history'][0]['loss_hartree']
        _pilot.atomic_json(directory / 'progress.json', dict(status='running',
            accepted_epoch=checkpoint['epoch'], target_epochs=epochs,
            loss_hartree=checkpoint['loss_hartree'], best_loss_hartree=best_loss,
            elapsed_seconds=time.perf_counter() - started))
        for epoch in range(checkpoint['epoch'] + 1, epochs + 1):
            attempted_epoch = epoch; tick = time.perf_counter()
            vector, state = optimizer.update(checkpoint['parameters'], checkpoint['gradient'],
                                             checkpoint['optimizer'])
            loss, gradient, records = evaluate(vector)
            record = dict(epoch=epoch, loss_hartree=loss, molecules=records)
            validation_history = checkpoint['validation_history']
            heldout_value = None
            if heldout is not None and (epoch % validation_interval == 0 or epoch == epochs):
                heldout_value, heldout_records = heldout(vector)
                validation_history = validation_history + [dict(epoch=epoch, loss_hartree=heldout_value, molecules=heldout_records)]
            # Save parameters only after measuring their own converged energy and gradient.
            checkpoint = dict(identity=identity, parameters=np.asarray(vector), optimizer=state,
                gradient=np.asarray(gradient), epoch=epoch, loss_hartree=loss, molecules=records,
                history=checkpoint['history'] + [record], validation_history=validation_history)
            save_checkpoint(last, checkpoint, optimizer)
            if loss < best_loss:
                save_checkpoint(best, checkpoint, optimizer); best_loss = loss
            if heldout_value is not None and heldout_value < best_validation_loss:
                save_checkpoint(validation_best, checkpoint, optimizer); best_validation_loss = heldout_value
            _pilot.atomic_json(directory / 'history.json', checkpoint['history'])
            if heldout is not None:
                _pilot.atomic_json(directory / 'validation_history.json', checkpoint['validation_history'])
            progress = dict(status='running', accepted_epoch=epoch, target_epochs=epochs,
                loss_hartree=loss, best_loss_hartree=best_loss,
                mean_energy_decrease_millihartree=1000 * (initial_loss - loss),
                epoch_elapsed_seconds=time.perf_counter() - tick,
                elapsed_seconds=time.perf_counter() - started)
            if heldout_value is not None:
                progress.update(validation_epoch=epoch, validation_loss_hartree=heldout_value,
                                best_validation_loss_hartree=best_validation_loss)
            if hasattr(optimizer, 'learning_rate_at'):
                progress['learning_rate'] = optimizer.learning_rate_at(epoch - 1)
            _pilot.atomic_json(directory / 'progress.json', progress)
            print(f"Epoch {epoch}/{epochs}: mean E {loss:.12f} Ha; "
                  f"decrease {progress['mean_energy_decrease_millihartree']:.6f} mHa; "
                  f"{progress['epoch_elapsed_seconds']:.2f} s", flush=True)
        summary = dict(status='complete', accepted_epoch=checkpoint['epoch'], target_epochs=epochs,
            loss_hartree=checkpoint['loss_hartree'], best_loss_hartree=best_loss,
            initial_loss_hartree=initial_loss, elapsed_seconds=time.perf_counter() - started,
            mean_energy_decrease_millihartree=1000 * (initial_loss - checkpoint['loss_hartree']),
            molecules=checkpoint['molecules'], identity=identity)
        if heldout is not None:
            summary.update(validation_history=checkpoint['validation_history'],
                           best_validation_loss_hartree=best_validation_loss)
    except Exception as error:
        summary = dict(status='failed', accepted_epoch=checkpoint['epoch'] if checkpoint else None,
            attempted_epoch=attempted_epoch, target_epochs=epochs, error=str(error),
            traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started,
            identity=identity)
        if vector is not None:
            snapshot = directory / f'failed-candidate-epoch-{attempted_epoch:06d}.npz'
            try:
                save_failure_snapshot(snapshot, vector, error, summary)
                summary['failure_snapshot'] = snapshot.name
            except Exception as snapshot_error:
                summary['failure_snapshot_error'] = str(snapshot_error)
    _pilot.atomic_json(directory / 'summary.json', summary)
    _pilot.atomic_json(directory / 'progress.json', summary)
    return summary


class Adam:
    def __init__(self, learning_rate, *, final_learning_rate=None, epochs=1000):
        import jax
        import optax
        from flax import serialization
        self.jax = jax; self.serialization = serialization
        if final_learning_rate is not None:
            if not 0 < final_learning_rate <= learning_rate or epochs < 1:
                raise ValueError('Cosine learning rates require 0 < final <= initial and positive epochs.')
            self.schedule = optax.cosine_decay_schedule(learning_rate, epochs,
                                                       alpha=final_learning_rate / learning_rate)
        else:
            self.schedule = lambda count: learning_rate
        self.transform = optax.chain(optax.clip_by_global_norm(1.), optax.adam(
            self.schedule if final_learning_rate is not None else learning_rate))
        self.step = jax.jit(self._step)

    def learning_rate_at(self, count):
        return float(self.schedule(count))

    def init(self, vector):
        return self.transform.init(self.jax.numpy.asarray(vector))

    def _step(self, vector, gradient, state):
        import optax
        updates, state = self.transform.update(gradient, state, vector)
        return optax.apply_updates(vector, updates), state

    def update(self, vector, gradient, state):
        return self.step(self.jax.numpy.asarray(vector), self.jax.numpy.asarray(gradient), state)

    def encode(self, state):
        return self.serialization.to_bytes(self.jax.device_get(state))

    def decode(self, data, template):
        return self.serialization.from_bytes(template, data)


def make_evaluators(rows, args, jax, experiment_class):
    from flax import nnx
    from jax.flatten_util import ravel_pytree
    from gradscf.data.molecule import atomic_number
    from gradscf.model.nnao import MACEBasisModel, build_graph
    model_config = dict(elements=(1, 6, 7, 8, 9), channels=8, num_interactions=2,
        max_ell=1, correlation=2, zero_init=True, basis_family='szp663_direct')
    model = MACEBasisModel(**model_config, rngs=nnx.Rngs(args.seed))
    import mace_jax
    vendor = Path(__file__).resolve().parents[1] / 'src' / 'gradscf' / 'model' / 'nnao'
    if not Path(mace_jax.__file__).resolve().is_relative_to(vendor):
        raise RuntimeError(f'MACE source mismatch: imported {mace_jax.__file__}; expected {vendor}.')
    definition, parameters, other = nnx.split(model, nnx.Param, ...)
    initial, unravel = ravel_pytree(parameters)
    if str(initial.dtype) != 'float64':
        raise RuntimeError('MACE parameter vector must use float64.')

    @jax.jit
    def predict(vector, graph):
        return nnx.merge(definition, unravel(vector), other)(graph)

    @jax.jit
    def backward(vector, graph, cotangent):
        return jax.vjp(lambda value: predict(value, graph), vector)[1](cotangent)[0]

    evaluators = []; value_evaluators = []; manifests = {}
    for row in rows:
        path = _pilot.cache_location(args.cache_dir, row)
        manifest = json.loads(path.with_suffix('.json').read_text())
        if _pilot.file_hash(path) != manifest['cache_sha256']:
            raise ValueError(f"{row['structure_id']}: native DF cache hash differs from its manifest.")
        experiment = experiment_class(geometry=row, basis_family='szp663_direct',
            core_primitives=6, jk_backend='df', auxbasis=args.auxbasis, integral_cache=path,
            implicit_tolerance=args.implicit_tolerance, cache_storage=args.cache_storage,
            scf_rescue_level_shift=args.scf_rescue_level_shift)
        graph = build_graph([atomic_number(symbol) for symbol in experiment.symbols],
                            experiment.coords, element_order=model.elements)
        if str(experiment.ps.dtype) != 'float64':
            raise RuntimeError('RHF integral arrays must use float64.')
        def evaluate(vector, graph=graph, experiment=experiment):
            vector = jax.numpy.asarray(vector)
            prediction = predict(vector, graph)
            value, output_gradient, details = experiment.evaluate(prediction)
            gradient = np.asarray(backward(vector, graph, output_gradient))
            return value, gradient, details
        forward = jax.jit(experiment._implicit_value)
        def value_only(vector, graph=graph, experiment=experiment, forward=forward):
            prediction = predict(jax.numpy.asarray(vector), graph)
            value, info = forward(prediction, experiment.ps, experiment.ph, experiment.rep)
            details = {key: bool(item) if key == 'converged' else int(item) if key == 'scf_cycles' else float(item)
                       for key, item in info.items()}
            return float(value), details
        evaluators.append((row['structure_id'], evaluate))
        value_evaluators.append((row['structure_id'], value_only))
        manifests[row['structure_id']] = dict(cache_sha256=manifest['cache_sha256'],
            integral_signature=experiment.integral_signature, nao=int(experiment.layout.topology.nao),
            primitive_nao=int(experiment.primitive_topology.nao), df_shape=list(experiment.rep.shape))
    return initial, evaluators, value_evaluators, manifests, model_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--validation-summary', type=Path, required=True)
    parser.add_argument('--heldout-structures', type=Path)
    parser.add_argument('--heldout-validation-summary', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--final-learning-rate', type=float)
    parser.add_argument('--validation-interval', type=int, default=20)
    parser.add_argument('--cache-storage', choices=('device', 'host', 'mmap'), default='device')
    parser.add_argument('--implicit-tolerance', type=float, default=1e-8)
    parser.add_argument('--scf-rescue-level-shift', type=float, default=0.0)
    parser.add_argument('--auxbasis', default='def2-universal-jkfit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.scf_rescue_level_shift) or args.scf_rescue_level_shift < 0:
        parser.error('SCF rescue level shift must be finite and nonnegative.')
    if (args.heldout_structures is None) != (args.heldout_validation_summary is None):
        parser.error('--heldout-structures and --heldout-validation-summary must be provided together.')
    if (args.epochs < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not np.isfinite(args.implicit_tolerance) or args.implicit_tolerance <= 0):
        parser.error('Epochs, learning rate and implicit tolerance must be positive and finite.')
    if args.validation_interval < 1 or (args.final_learning_rate is not None and (
            not np.isfinite(args.final_learning_rate) or not 0 < args.final_learning_rate <= args.learning_rate)):
        parser.error('Validation interval must be positive; final learning rate must be finite and in (0, initial].')
    rows = _pilot.read_structures(args.structures)
    heldout_rows = _pilot.read_structures(args.heldout_structures) if args.heldout_structures else []
    if set(row['structure_id'] for row in rows) & set(row['structure_id'] for row in heldout_rows):
        raise ValueError('Train and held-out structure IDs overlap.')
    validation = verify_validation(args.validation_summary, args.structures,
        [row['structure_id'] for row in rows], tolerance=args.implicit_tolerance,
        rescue_level_shift=args.scf_rescue_level_shift)
    root = Path(__file__).resolve().parents[1]
    source_hash = _pilot.scientific_source_hash(root)
    verify_validation_sources(validation, root, scientific_hash=source_hash)
    heldout_validation = None
    if heldout_rows:
        heldout_validation = verify_validation(args.heldout_validation_summary, args.heldout_structures,
            [row['structure_id'] for row in heldout_rows], tolerance=args.implicit_tolerance,
            rescue_level_shift=args.scf_rescue_level_shift)
        verify_validation_sources(heldout_validation, root, scientific_hash=source_hash)
    jax, experiment_class = _pilot.numerical_runtime('cuda')
    initial, evaluators, value_evaluators, manifests, model_config = make_evaluators(rows + heldout_rows, args, jax, experiment_class)
    for record in validation['results'] + (heldout_validation['results'] if heldout_validation else []):
        if manifests[record['structure_id']]['cache_sha256'] != record.get('cache_sha256'):
            raise ValueError(f"{record['structure_id']}: training DF cache differs from validation.")
    identity = dict(structures_sha256=_pilot.file_hash(args.structures),
        structure_ids=[row['structure_id'] for row in rows], caches=manifests,
        scientific_source_sha256=source_hash, source_sha256=_pilot.file_hash(__file__),
        experiment_source_sha256=_pilot.file_hash(Path(__file__).with_name('optimize_methane_nnao.py')),
        validation_source_sha256=_pilot.file_hash(Path(__file__).with_name('validate_nnao_pilot.py')),
        validation_summary_sha256=_pilot.file_hash(args.validation_summary),
        model_config=json.loads(json.dumps(model_config)), seed=args.seed,
        parameter_count=int(initial.size), method='RHF', jk_backend='df', core_primitives=6,
        auxbasis=args.auxbasis, dtype='float64', implicit_tolerance=args.implicit_tolerance,
        scf_rescue_level_shift=args.scf_rescue_level_shift,
        optimizer='Optax clip_by_global_norm(1.0) + Adam', learning_rate=args.learning_rate,
        loss='equal-weight mean of all selected reconverged total energies in Hartree',
        scf_config=dict(max_cycle=150, conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9),
        implicit_config=dict(tolerance=args.implicit_tolerance, max_iter=100, restart=40))
    identity.update(cache_storage=args.cache_storage, final_learning_rate=args.final_learning_rate,
        learning_rate_schedule='cosine' if args.final_learning_rate is not None else 'constant',
        learning_rate_decay_updates=args.epochs if args.final_learning_rate is not None else None,
        initialization='fresh seeded MACE with trainable reference bias; no earlier weights',
        heldout_structure_ids=[row['structure_id'] for row in heldout_rows],
        heldout_structures_sha256=_pilot.file_hash(args.heldout_structures) if heldout_rows else None,
        heldout_validation_summary_sha256=_pilot.file_hash(args.heldout_validation_summary) if heldout_rows else None,
        validation_interval=args.validation_interval if heldout_rows else None,
        gradient_batch='full train dataset, equal mean, one Adam update per epoch')
    from importlib.metadata import version
    metadata = dict(identity=identity, target_epochs=args.epochs, device_kinds=[d.device_kind for d in jax.devices()],
        devices=[str(d) for d in jax.devices()], backend=jax.default_backend(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), imported_sources=_pilot.verify_source_imports(root),
        python_version=platform.python_version(), hostname=socket.gethostname(),
        parameter_dtype=str(initial.dtype),
        train_count=len(rows), heldout_count=len(heldout_rows),
        fixed_cache_storage={'device':'all fixed caches on GPU',
            'host':'full NumPy arrays on CPU; NPZ is not memory mapped',
            'mmap':'read-only numeric memmap views of original ZIP_STORED NPZ; OS can reclaim file-backed pages; current structure transferred to GPU'}[args.cache_storage],
        mace_source=str(Path(__import__('mace_jax').__file__).resolve()),
        versions={name: version(name) for name in ('jax', 'jaxlib', 'flax', 'optax', 'e3nn-jax', 'mace-jax')})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _pilot.atomic_json(args.output_dir / 'metadata.json', metadata)
    def evaluate(vector):
        _pilot.atomic_json(args.output_dir / 'batch_progress.json', dict(completed=0, total=len(rows), molecules=[]))
        return evaluate_batch(vector, evaluators[:len(rows)], progress=lambda records: _pilot.atomic_json(
            args.output_dir / 'batch_progress.json', dict(completed=len(records), total=len(rows), molecules=records)))
    def heldout(vector):
        _pilot.atomic_json(args.output_dir / 'heldout_progress.json', dict(completed=0, total=len(heldout_rows), molecules=[]))
        return evaluate_values(vector, value_evaluators[len(rows):], progress=lambda records: _pilot.atomic_json(
            args.output_dir / 'heldout_progress.json', dict(completed=len(records), total=len(heldout_rows), molecules=records)))
    summary = run_training(initial, evaluate, Adam(args.learning_rate,
        final_learning_rate=args.final_learning_rate, epochs=args.epochs), args.output_dir,
        identity, epochs=args.epochs, resume=args.resume,
        heldout=heldout if heldout_rows else None, validation_interval=args.validation_interval)
    return int(summary['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
