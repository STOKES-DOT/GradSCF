"""Measure the actual shared-trainer path for one existing validated structure."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import numpy as np
import jax
import train_nnao_pilot as trainer
import validate_nnao_pilot as pilot
from optimize_methane_nnao import MethaneRHF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure', default='S22/07')
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(structure_id=args.structure, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        gradient='actual shared-trainer MACE parameter VJP through implicit RHF',
        jax_version=jax.__version__, numpy_version=np.__version__,
        preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        memory_fraction=os.environ.get('XLA_CLIENT_MEM_FRACTION'),
        allocator=os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR'), timings=[])
    stop = threading.Event()
    samples = []

    def sample_memory():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    fields = [x.strip() for x in line.split(',')]
                    if len(fields) == 2 and fields[0] == str(os.getpid()):
                        try:
                            samples.append(int(fields[1]))
                        except ValueError:
                            pass
            stop.wait(.2)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        device = jax.devices()[0]
        if jax.default_backend() != 'gpu':
            raise RuntimeError('This probe must run on its allocated GPU.')
        report.update(device=str(device), device_kind=device.device_kind,
            imported_sources=pilot.verify_source_imports(ROOT),
            scientific_source_sha256=pilot.scientific_source_hash(ROOT),
            software_versions={p:version(p) for p in ('jaxlib', 'flax', 'optax', 'mace-jax', 'e3nn-jax')})
        data = ROOT / 'artifacts/nnao-stage2-20261007/selection/all.jsonl'
        row = next(r for r in map(json.loads, data.read_text().splitlines())
                   if r['structure_id'] == args.structure)
        report.update(natoms=row['natoms'], formula=row['formula_group'], dimensions=row['size_estimate'])
        setup = argparse.Namespace(seed=0, auxbasis='def2-universal-jkfit',
            cache_dir=args.cache_dir, cache_storage='mmap', implicit_tolerance=1e-8,
            scf_rescue_level_shift=.2)
        initial, evaluators, _, manifests, model = trainer.make_evaluators([row], setup, jax, MethaneRHF)
        report.update(model_config=model, parameter_count=int(initial.size), manifests=manifests,
            allocator_before=device.memory_stats())
        for index in range(3):
            tick = time.perf_counter()
            energy, gradient, info = evaluators[0][1](initial)
            if not np.isfinite(gradient).all() or not trainer.valid_scf(info):
                raise RuntimeError('Invalid full implicit-gradient evaluation.')
            report['timings'].append(dict(index=index, elapsed_seconds=time.perf_counter()-tick,
                energy_hartree=energy, parameter_gradient_norm=float(np.linalg.norm(gradient)),
                scf=info, allocator=device.memory_stats()))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        stop.set(); sampler.join(timeout=5)
        report.update(sampled_process_peak_mib=max(samples, default=None),
            process_memory_sample_count=len(samples),
            sampled_peak_note='Own-process nvidia-smi polling every 0.2s; transient peaks may be missed. '
                'Allocator peak_bytes_in_use is exact only for allocations tracked by JAX.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
