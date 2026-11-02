"""Stage C: train complete MACE contractions at a selected fixed primitive pool."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from ..runtime import (atomic_json, cache_location, repository_root,
                       scientific_source_hash, verify_source_imports)
from .pool_calibration import _measure, _validate_definition, load_references


def run_fixed_pool(data, candidate, cache_dir, output_dir, *, epochs=5,
                   learning_rate=1e-4, seed=0, storage='device', resume=False):
    """Short validated full-batch training; one epoch updates all shared MACE weights.

    Candidate exponents remain external immutable data, not optimizer variables.
    No exponent pullback is built. Resume checks identity and re-evaluates the
    accepted energy/gradient before allowing the saved Adam state to advance.
    """
    _validate_definition(data)
    rows = data['structures']
    if not rows or any(row.get('weight') != 1 for row in rows):
        raise ValueError('Stage C pilot currently requires explicit equal unit weights.')
    if len({row['structure_id'] for row in rows}) != len(rows):
        raise ValueError('Structure IDs must be unique.')
    if epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Epochs and learning rate must be positive.')
    output = Path(output_dir)
    if (output/'last.npz').exists() != resume:
        raise ValueError('Use a fresh output directory, or --resume with an existing last.npz.')
    import jax
    import jax.numpy as jnp
    from gradscf.data.molecule import atomic_number
    from ..basis import prepare_direct_basis
    from ..config import MODEL_CONFIG
    from ..graph import build_graph
    from ..initialization import bind_pool_candidate
    from ..integrals.fixed_pool import prepare_fixed_pool
    from ..training.checkpoints import load_checkpoint
    from ..training.trainer import (Adam, evaluate_batch,
                                    make_mace_predictor, run_training)
    if not jax.config.jax_enable_x64:
        raise ValueError('Enable JAX float64 before constructing the model.')
    started = time.perf_counter()
    measurements, manifests, evaluators, value_evaluators = [], {}, [], []
    initial, predict, backward = _measure('prepare_mace',
        lambda: make_mace_predictor(seed, MODEL_CONFIG), measurements)
    for row in rows:
        identifier = row['structure_id']
        with jax.default_device(jax.devices('cpu')[0]):
            layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                cart=False, basis_family='szp663_direct', core_primitives=6)
            layout = bind_pool_candidate(layout, candidate)
        experiment, manifest = _measure(f'{identifier}/prepare_or_load_fixed_cache',
            lambda: prepare_fixed_pool(row, layout, cache_location(cache_dir, row),
                                       storage=storage), measurements)
        manifests[identifier] = manifest
        graph = build_graph([atomic_number(s) for s in row['symbols']], row['coords_angstrom'],
                            element_order=MODEL_CONFIG['elements'])
        # Zero head kernels must initialize the full raw reference contraction.
        expected = np.array(layout.reference_outputs(), copy=True)
        expected /= np.maximum(np.linalg.norm(expected, axis=-1, keepdims=True), 1e-30)
        np.testing.assert_allclose(predict(initial, graph), expected, atol=1e-13, rtol=1e-13)
        def evaluate(vector, graph=graph, experiment=experiment):
            tick = time.perf_counter()
            prediction = predict(jnp.asarray(vector), graph)
            energy, coefficient_gradient, info = experiment.evaluate(prediction)
            gradient = np.asarray(backward(jnp.asarray(vector), graph, coefficient_gradient))
            # Batch control supplies the canonical ID and energy fields.
            info = {key: value for key, value in info.items()
                    if key not in {'structure_id', 'energy_hartree'}}
            return energy, gradient, dict(info, full_backward_seconds=time.perf_counter()-tick)
        def value_only(vector, graph=graph, experiment=experiment):
            energy, _, info = experiment.evaluate(predict(jnp.asarray(vector), graph), gradient=False)
            return energy, info
        evaluators.append((identifier, evaluate))
        value_evaluators.append((identifier, value_only))
    identity = json.loads(json.dumps(dict(stage='C', trainable='mace_parameters_only',
        candidate=candidate, references=data, caches=manifests,
        scientific_source_sha256=scientific_source_hash(repository_root()),
        model_config=MODEL_CONFIG, seed=seed, learning_rate=learning_rate,
        optimizer='clip_by_global_norm(1) + Adam', parameter_count=int(initial.size),
        initialization_sha256=hashlib.sha256(np.asarray(initial).tobytes()).hexdigest(),
        loss='equal mean of converged all-electron RHF total energies / Hartree',
        method='RHF', jk_backend='df', auxbasis='def2-universal-jkfit',
        scf=dict(max_cycle=150, conv_tol=1e-12, conv_tol_density=1e-10,
                 conv_tol_grad=1e-9, rescue_level_shift=.2),
        implicit=dict(tolerance=1e-8, max_iter=100, restart=40),
        storage=storage, dtype='float64', jax=jax.__version__, backend=jax.default_backend())))
    optimizer = Adam(learning_rate)
    saved = load_checkpoint(output/'last.npz', identity, optimizer, initial) if resume else None
    vector = saved['parameters'] if saved else initial
    def batch(vector):
        return evaluate_batch(vector, evaluators, progress=lambda records:
            atomic_json(output/'batch_progress.json', dict(completed=len(records),
                        total=len(rows), molecules=records)))
    cold = _measure('first_full_batch_including_compilation', lambda: batch(vector), measurements)
    hot = _measure('hot_full_batch', lambda: batch(vector), measurements)
    np.testing.assert_allclose(hot[0], cold[0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(hot[1], cold[1], atol=1e-9, rtol=1e-8)
    if saved:
        np.testing.assert_allclose(hot[0], saved['loss_hartree'], atol=1e-9, rtol=0)
        np.testing.assert_allclose(hot[1], saved['gradient'], atol=1e-7, rtol=1e-5)
    direction = np.random.default_rng(seed).normal(size=initial.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-4
    checks = []
    for (identifier, evaluate), (_, forward) in zip(evaluators, value_evaluators):
        _, gradient, _ = evaluate(vector)
        plus, _ = forward(vector+step*direction)
        minus, _ = forward(vector-step*direction)
        ad, fd = float(gradient@direction), (plus-minus)/(2*step)
        checks.append(dict(structure_id=identifier, ad=ad, fd=fd, step=step,
            atol=2e-6, rtol=2e-5, passed=bool(np.isclose(ad, fd, atol=2e-6, rtol=2e-5))))
    passed = all(check['passed'] for check in checks)
    atomic_json(output/'gate.json', dict(passed=passed, parameter_direction_checks=checks,
        hot_repeat_passed=True, resume_replay_passed=True if saved else None,
        native_derivatives_in_training=False))
    if not passed:
        raise RuntimeError('MACE-to-implicit-SCF parameter gradient gate failed.')
    atomic_json(output/'metadata.json', dict(identity=identity, target_epochs=epochs,
        devices=[str(d) for d in jax.devices()], device_kinds=[d.device_kind for d in jax.devices()],
        host=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        imported_sources=verify_source_imports(repository_root()),
        preparation_measurements=measurements))
    summary = _measure('training', lambda: run_training(initial, batch, optimizer,
        output, identity, epochs=epochs, resume=resume), measurements)
    atomic_json(output/'timings.json', dict(measurements=measurements,
        wall_seconds=time.perf_counter()-started))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--storage', choices=('host', 'device', 'mmap'), default='device')
    parser.add_argument('--platform', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    os.environ['JAX_ENABLE_X64'] = '1'
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu' if args.platform == 'cuda' else 'cpu'
    import jax
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != ('gpu' if args.platform == 'cuda' else 'cpu'):
        raise RuntimeError('Requested JAX backend is unavailable.')
    result = run_fixed_pool(load_references(args.references), json.loads(args.candidate.read_text()),
        args.cache_dir, args.output_dir, epochs=args.epochs, learning_rate=args.learning_rate,
        seed=args.seed, storage=args.storage, resume=args.resume)
    return int(result['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
