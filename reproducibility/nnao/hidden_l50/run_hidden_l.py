"""Matched hidden-L experiment; reuse the production fixed-pool and trainer APIs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np


HERE = Path(__file__).absolute().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('L1', 'L2'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan = json.loads((HERE/'execution-plan.json').read_text())
    profile = plan['profiles'][args.profile]
    output = HERE/'runs'/args.profile

    import jax
    import jax.numpy as jnp
    from gradscf.model.nnao.basis import prepare_direct_basis
    from gradscf.model.nnao.graph import build_graph
    from gradscf.model.nnao.initialization import bind_pool_candidate
    from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
    from gradscf.model.nnao.runtime import (atomic_json, cache_location, file_hash,
        repository_root, scientific_source_hash, verify_source_imports)
    from gradscf.model.nnao.training.trainer import (Adam, evaluate_batch,
        make_mace_predictor, run_training)
    from gradscf.model.nnao.workflows.pool_calibration import load_references

    record = dict(profile=args.profile, status='preparing',
        started_at=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), host=platform.node())
    execution = output/'execution.json'
    atomic_json(execution, record)
    started = time.perf_counter()
    try:
        if not jax.config.jax_enable_x64 or jax.default_backend() != 'gpu':
            raise RuntimeError('Require JAX float64 and CUDA; no backend fallback.')
        devices = jax.devices('gpu')
        if len(devices) != 1 or 'A100' not in devices[0].device_kind:
            raise RuntimeError(f'Require one Slurm-assigned A100: {devices}')
        if '--xla_gpu_enable_command_buffer=' not in os.environ.get('XLA_FLAGS', '').split():
            raise RuntimeError('Disable CUDA command buffers before importing JAX.')
        root = repository_root()
        if scientific_source_hash(root) != plan['scientific_source_sha256']:
            raise RuntimeError('Pinned scientific source changed.')
        sources = verify_source_imports(root)
        for name, expected in plan['input_sha256'].items():
            if file_hash(HERE/name) != expected:
                raise RuntimeError(f'Input changed: {name}')
        data = load_references(HERE/'inputs/references.json')
        candidate = json.loads((HERE/'inputs/candidate.json').read_text())
        config = profile['model_config']
        initial, predict, backward, mask = make_mace_predictor(0, config, return_mask=True)
        preparation = dict(model_seconds=time.perf_counter()-started)
        manifests, evaluators, initial_hash = {}, [], hashlib.sha256()
        for row in data['structures']:
            tick = time.perf_counter()
            name = row['structure_id']
            path = cache_location(plan['cache_root'], row)
            # Never regenerate a missing cache silently in this comparison.
            if not path.is_file() or not path.with_suffix('.json').is_file():
                raise FileNotFoundError(f'Missing matched DZP-M cache: {path}')
            with jax.default_device(jax.devices('cpu')[0]):
                layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                    cart=False, basis_family=data['basis_family'], core_primitives=data['core_primitives'])
                layout = bind_pool_candidate(layout, candidate)
                expected = np.array(layout.reference_outputs(), copy=True)
                expected /= np.maximum(np.linalg.norm(expected, axis=-1, keepdims=True), 1e-30)
            initial_hash.update(name.encode())
            initial_hash.update(expected.tobytes())
            experiment, manifest = prepare_fixed_pool(row, layout, path, storage='mmap')
            if manifest['cache_sha256'] != plan['caches'][name]['cache_sha256']:
                raise RuntimeError(f'Cache differs from source checkpoint: {name}')
            manifests[name] = manifest
            graph = build_graph(row['atomic_numbers'], row['coords_angstrom'],
                element_order=config['elements'], cutoff=config['cutoff'])

            def evaluate(vector, graph=graph, experiment=experiment, expected=expected):
                start = time.perf_counter()
                vector = jnp.asarray(vector)
                prediction = predict(vector, graph)
                if not args.resume and evaluate_first_pass[0]:
                    np.testing.assert_allclose(prediction, expected, atol=1e-13, rtol=1e-13)
                energy, cotangent, info = experiment.evaluate_device(prediction)
                gradient = np.asarray(backward(vector, graph, cotangent))
                details = {k: v for k, v in info.items() if k not in {'structure_id', 'energy_hartree'}}
                return energy, gradient, dict(details, full_backward_seconds=time.perf_counter()-start)

            evaluators.append((name, evaluate))
            preparation[name] = dict(seconds=time.perf_counter()-tick,
                contracted_nao=layout.topology.nao, cache_bytes=manifest['bytes'])
            atomic_json(output/'preparation.json', dict(completed=len(evaluators),
                total=len(data['structures']), measurements=preparation))
        identity = dict(stage='C-hidden-L-comparison', trainable='mace_parameters_only',
            model_config=config, references=data, candidate=candidate, caches=manifests,
            seed=0, optimizer=plan['optimizer'], scientific_source_sha256=plan['scientific_source_sha256'],
            parameter_count=int(initial.size), trainable_parameter_count=int(mask.sum()),
            frozen_parameter_indices=np.flatnonzero(~mask).tolist(),
            initialization_sha256=hashlib.sha256(np.asarray(initial).tobytes()).hexdigest(),
            initial_coefficients_sha256=initial_hash.hexdigest(),
            scf=plan['scf'], implicit=plan['implicit'], method='RHF', jk_backend='df',
            auxbasis='def2-universal-jkfit', dtype='float64', storage='mmap',
            backend=jax.default_backend(), jax=jax.__version__,
            loss='equal mean of converged all-electron RHF total energies / Hartree')
        atomic_json(output/'metadata.json', dict(identity=identity, device_kind=devices[0].device_kind,
            imported_sources=sources, preparation_seconds=time.perf_counter()-started,
            target_epochs=plan['epochs'], finite_differences=False, extra_scf_evaluations=False))
        optimizer = Adam(plan['optimizer']['learning_rate'],
            final_learning_rate=plan['optimizer']['final_learning_rate'],
            epochs=plan['optimizer']['cosine_updates'], trainable_mask=mask)
        evaluate_first_pass = [True]

        def batch(vector):
            result = evaluate_batch(vector, evaluators, progress=lambda rows:
                atomic_json(output/'batch_progress.json', dict(completed=len(rows),
                    total=len(evaluators), molecules=rows)))
            evaluate_first_pass[0] = False
            atomic_json(output/'memory.json', dict(
                peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                gpu_allocator=devices[0].memory_stats(),
                elapsed_seconds=time.perf_counter()-started))
            return result

        record['status'] = 'training'
        atomic_json(execution, record)
        result = run_training(initial, batch, optimizer, output, identity,
            epochs=plan['epochs'], resume=args.resume)
        record.update(status=result['status'], accepted_epoch=result['accepted_epoch'])
        if result['status'] != 'complete':
            raise RuntimeError(result.get('error', 'Training did not complete.'))
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        record.update(ended_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter()-started,
            peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total',
            '--format=csv,noheader'], capture_output=True, text=True)
        record['gpu_memory_at_exit'] = gpu.stdout.strip()
        atomic_json(execution, record)


if __name__ == '__main__':
    main()
