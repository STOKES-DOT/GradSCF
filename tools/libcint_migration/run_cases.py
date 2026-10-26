"""Serial bounded subprocess runner with sampled process-group RSS."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def processes():
    values = {}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            stat = (p/'stat').read_text().split(') ', 1)[1].split()
            values[int(p.name)] = (int(stat[1]), int(stat[2]), int(stat[21])*os.sysconf('SC_PAGE_SIZE'))
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            pass
    return values


def run(config, root, *, deadline, timeout=6000, profile=False):
    case = root/'cases'/config['case_id']
    case.mkdir(parents=True, exist_ok=True)
    path = case/'input.json'
    path.write_text(json.dumps(config, indent=2)+'\n')
    output = root/'runs'/config['case_id']
    output.mkdir(parents=True, exist_ok=True)
    entry = 'probes.py' if 'probe' in config else 'worker.py'
    command = [sys.executable, str(Path(__file__).with_name(entry)), str(path), str(output)]
    if profile:
        command.append('--profile')
    env = dict(os.environ)
    threads = str(config['threads'])
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = threads
    env.update(JAX_PLATFORMS='cuda,cpu' if config.get('device') == 'gpu' else 'cpu',
               JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false',
               XLA_PYTHON_CLIENT_PREALLOCATE='false')
    affinity = sorted(os.sched_getaffinity(0))[:config['threads']]
    def restrict():
        os.sched_setaffinity(0, affinity)
    t0 = time.perf_counter()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stop = min(time.time()+timeout, deadline)
    peak, samples = 0, 0
    timeline = []
    next_progress = 0.
    status = 'ok'
    with (output/'stdout.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, preexec_fn=restrict)
        while child.poll() is None:
            stats = processes()
            rss = sum(v[2] for v in stats.values() if v[1] == child.pid)
            peak = max(peak, rss)
            samples += 1
            timeline.append([time.perf_counter(), rss])
            elapsed = time.perf_counter()-t0
            if elapsed >= next_progress:
                # Controller-only OS telemetry; no callback, profiler or extra
                # numerical evaluation enters the measured worker.
                progress = dict(case_id=config['case_id'], pid=child.pid,
                    elapsed_seconds=elapsed, current_group_rss_bytes=rss,
                    peak_group_rss_bytes=peak, status='running')
                try:
                    stat = Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()
                    progress['worker_cpu_seconds'] = (int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK')
                except (FileNotFoundError, ProcessLookupError):
                    pass
                (output/'progress.json').write_text(json.dumps(progress)+'\n')
                next_progress = elapsed+5.
            if time.time() > stop:
                status = 'timeout'
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(.05)
        code = child.wait()
    result = dict(case_id=config['case_id'], engine=config['engine'], config=config, command=command,
        process_wall_seconds=time.perf_counter()-t0, returncode=code,
        status=status if status != 'ok' else ('ok' if code == 0 else 'error'),
        peak_group_rss_bytes=peak, rss_samples=samples, rss_sampling_seconds=.05,
        memory_method='sum resident Linux /proc stat RSS for child process group, sampled at 50ms; short peaks can be missed',
        started_utc=started_utc, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stdout=str(output/'stdout.log'), affinity=affinity, instrumented=profile)
    (output/'rss-timeline.json').write_text(json.dumps(timeline)+'\n')
    if (output/'records.json').is_file():
        result['records'] = json.loads((output/'records.json').read_text())
        if result['status'] == 'ok' and any(r['status'] != 'ok' for r in result['records']):
            result['status'] = result['records'][-1]['status']
    (output/'process.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'progress.json').write_text(json.dumps(dict(
        case_id=config['case_id'], status=result['status'],
        elapsed_seconds=result['process_wall_seconds'], peak_group_rss_bytes=peak))+'\n')
    with (root/'results.jsonl').open('a') as stream:
        stream.write(json.dumps(result)+'\n')
    print(config['case_id'], result['status'], round(result['process_wall_seconds'], 3), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('matrix', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=int, default=6300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.time()+args.seconds
    for config in json.loads(args.matrix.read_text()):
        if time.time()+15 >= deadline:
            print('Global budget reached; unstarted cases remain unrun.', flush=True)
            break
        run(config, args.output, deadline=deadline, timeout=config.get('timeout', 6000), profile=config.get('profile', False))


if __name__ == '__main__':
    main()
