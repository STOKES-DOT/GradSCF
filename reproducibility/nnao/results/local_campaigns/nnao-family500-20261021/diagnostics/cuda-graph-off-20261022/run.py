"""Bounded CUDA-command-buffer comparison using an immutable failed-run seed."""
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent.parent
OUTPUT = ROOT / 'run'
FAMILY = 'nnao_tzp_s'
MAX_UPDATES = 40


def read(path):
    return json.loads(path.read_text())


def main():
    flags = os.environ.get('XLA_FLAGS', '').split()
    if '--xla_gpu_enable_command_buffer=' not in flags:
        raise RuntimeError('Command buffers must be disabled before importing JAX.')
    # Scientific code and campaign inputs remain unchanged; only runtime capture differs.
    spec = importlib.util.spec_from_file_location('family500_campaign', CAMPAIGN / 'campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    campaign.verify()
    import jax
    from gradscf.model.nnao.runtime import atomic_json, file_hash
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    if jax.default_backend() != 'gpu' or len(jax.devices('gpu')) != 1:
        raise RuntimeError('Exactly one Slurm-assigned GPU is required.')
    seed = read(ROOT / 'seed.json')
    checkpoint = OUTPUT / 'last.npz'
    if not (OUTPUT / 'progress.json').exists() and file_hash(checkpoint) != seed['checkpoint_sha256']:
        raise ValueError('Diagnostic checkpoint does not match its immutable seed.')
    data, model = campaign.configuration(FAMILY)
    candidate = read(CAMPAIGN / 'inputs/s-candidate.json')
    started = time.perf_counter()
    stopped = threading.Event()

    def observe():
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        with (ROOT / 'resources.jsonl').open('a', buffering=1) as stream:
            while not stopped.is_set():
                record = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    elapsed_seconds=time.perf_counter() - started,
                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                for name in ('progress.json', 'phase.json', 'batch_progress.json'):
                    path = OUTPUT / name
                    if path.exists():
                        record[name] = read(path)
                if visible:
                    query = ['nvidia-smi', '--id=' + visible,
                        '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits']
                    try:
                        result = subprocess.run(query, capture_output=True, text=True, timeout=5)
                        record['gpu_csv'] = result.stdout.strip()
                        if result.returncode:
                            record['gpu_query_error'] = result.stderr.strip()
                    except (OSError, subprocess.TimeoutExpired) as error:
                        record['gpu_query_error'] = str(error)
                stream.write(json.dumps(record) + '\n')
                stopped.wait(10)

    atomic_json(ROOT / 'execution.json', dict(status='running', family=FAMILY,
        source_update=seed['update_step'], target_update=MAX_UPDATES,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), xla_flags=os.environ['XLA_FLAGS'],
        devices=[d.device_kind for d in jax.devices('gpu')], jax=jax.__version__,
        finite_differences=False, extra_evaluations=False))
    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    try:
        result = run_fixed_pool(data, candidate, CAMPAIGN / 'cache/s', OUTPUT,
            epochs=1000, learning_rate=1e-3, seed=0, storage='mmap', resume=True,
            model_config=model, batch_size=50, decay_epochs=100,
            check_parameter_gradients=False, max_updates=MAX_UPDATES,
            early_stop_window=10, early_stop_tolerance=1e-4)
        if result['status'] != 'update_budget_reached' or result['update_step'] != MAX_UPDATES:
            raise RuntimeError('Diagnostic stopped before its required update budget.')
        atomic_json(ROOT / 'result.json', dict(status='passed_update_budget',
            update_step=result['update_step'], completed_epochs=result['epoch'],
            history=result['history'], checkpoint_sha256=file_hash(checkpoint),
            wall_seconds=time.perf_counter() - started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            original_failure_update=17, global_memory_stability_proven=False))
    except Exception as error:
        atomic_json(ROOT / 'result.json', dict(status='failed', error=str(error),
            wall_seconds=time.perf_counter() - started))
        raise
    finally:
        stopped.set()
        observer.join(timeout=7)


if __name__ == '__main__':
    main()
