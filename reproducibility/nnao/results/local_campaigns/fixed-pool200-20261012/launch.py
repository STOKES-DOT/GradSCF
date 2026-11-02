"""Prepare fixed caches in isolated CPU processes, then run the existing stage C."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def save(name, data):
    path = OUT/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def prepare_one(index, cpus):
    os.sched_setaffinity(0, set(map(int, cpus.split(','))))
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    jax.config.update('jax_enable_x64', True)
    from gradscf.model.nnao.workflows.pool_calibration import load_references
    from gradscf.model.nnao.basis import prepare_direct_basis
    from gradscf.model.nnao.initialization import bind_pool_candidate
    from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
    from gradscf.model.nnao.runtime import cache_location
    row = load_references(OUT/'references.json')['structures'][index]
    candidate = json.loads((OUT/'candidate-exponents.json').read_text())
    started = time.perf_counter()
    layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                                   cart=False, basis_family='szp663_direct', core_primitives=6)
    layout = bind_pool_candidate(layout, candidate)
    _, manifest = prepare_fixed_pool(row, layout, cache_location(OUT/'cache', row), storage='mmap')
    print(json.dumps(dict(structure_id=row['structure_id'], seconds=time.perf_counter()-started,
        bytes=manifest['bytes'], pool_nao=manifest['pool_nao'],
        contracted_nao=manifest['contracted_nao'], signature=manifest['signature'])), flush=True)


def launch():
    rows = json.loads((OUT/'references.json').read_text())['structures']
    assert len(rows) == 200 and len({r['structure_id'] for r in rows}) == 200
    budget = int(os.environ['SLURM_CPUS_PER_TASK'])
    cpus = sorted(os.sched_getaffinity(0))[:budget]
    assert 4 <= len(cpus) <= 64, cpus
    os.sched_setaffinity(0, set(cpus))
    started = time.perf_counter()
    records = []
    save('launch.json', dict(status='preparing_caches', count=200, completed=0,
        seed=0, initialization='fresh seeded MACE; no parent checkpoint', epochs=1000,
        learning_rate=1e-4, heldout_count=0, storage='mmap', cpu_affinity=cpus,
        vm_max_map_count=int(Path('/proc/sys/vm/max_map_count').read_text()),
        slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    logs = OUT/'prepare-logs'
    logs.mkdir(exist_ok=True)
    def shard(worker):
        selected_cpus = cpus[worker::4]
        for index in range(worker, len(rows), 4):
            row = rows[index]
            log = logs/(hashlib.sha256(row['structure_id'].encode()).hexdigest()[:16]+'.log')
            with log.open('w') as stream:
                result = subprocess.run([sys.executable, __file__, 'prepare', str(index),
                    ','.join(map(str, selected_cpus))], stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Cache preparation failed for {row['structure_id']}; see {log}")
            record = json.loads(log.read_text().splitlines()[-1])
            with lock:
                records.append(record)
                save('preparation.json', dict(status='preparing', completed=len(records),
                    total=len(rows), elapsed_seconds=time.perf_counter()-started, records=records))
                print(f"Cache {len(records)}/200: {row['structure_id']}; {record['bytes']/2**30:.3f} GiB", flush=True)
    import threading
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(shard, worker) for worker in range(4)]
        for future in futures:
            future.result()
    save('preparation.json', dict(status='complete', completed=200, total=200,
        elapsed_seconds=time.perf_counter()-started, bytes=sum(r['bytes'] for r in records), records=records))
    env = dict(os.environ, JAX_PLATFORMS='cuda,cpu')
    command = [sys.executable, '-u', '-m', 'gradscf.model.nnao.workflows.fixed_pool',
        '--platform', 'cuda', '--references', str(OUT/'references.json'),
        '--candidate', str(OUT/'candidate-exponents.json'), '--cache-dir', str(OUT/'cache'),
        '--output-dir', str(OUT/'training'), '--epochs', '1000', '--learning-rate', '1e-4',
        '--seed', '0', '--storage', 'mmap']
    if (OUT/'training/last.npz').exists():
        command.append('--resume')
    with (OUT/'training.log').open('a') as log, (OUT/'resource-watch.jsonl').open('a') as watch:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        save('launch.json', dict(status='scf_gate_then_training', pid=process.pid,
            command=command, slurm_job_id=os.environ.get('SLURM_JOB_ID')))
        import psutil
        while process.poll() is None:
            try:
                memory = psutil.Process(process.pid).memory_info()
                with Path(f'/proc/{process.pid}/maps').open() as stream:
                    maps = sum(1 for _ in stream)
                watch.write(json.dumps(dict(time=time.time(), rss_bytes=memory.rss,
                    vms_bytes=memory.vms, maps=maps))+'\n')
                watch.flush()
            except (ProcessLookupError, FileNotFoundError, psutil.NoSuchProcess):
                pass
            time.sleep(15)
    save('exit.json', dict(exit_code=process.returncode,
        elapsed_seconds=time.perf_counter()-started, slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    return process.returncode


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        prepare_one(int(sys.argv[2]), sys.argv[3])
    else:
        try:
            raise SystemExit(launch())
        except Exception as error:
            save('launch-failure.json', dict(error=str(error)))
            raise
