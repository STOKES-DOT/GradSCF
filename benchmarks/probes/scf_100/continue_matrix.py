"""Serial continuation of frozen SCF-100 rows inside one existing allocation."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

RSS_LIMIT = 200 * 1024**3


def read(path):
    return json.loads(path.read_text())


def configurations(root):
    plan = root/'baseline/reports/scf_100/20261105-40a569bf/plan-v2'
    order, manifest = read(plan/'run_plan.json'), read(plan/'input_manifest.json')
    cases, hashes = order['run_order'], manifest['config_file_sha256']
    if len(cases) != len(set(cases)) or {case+'.json' for case in cases} != set(hashes):
        raise ValueError('Frozen plan and configuration manifest differ')
    if len(cases) != manifest['row_count'] or order['rss_guard_bytes'] != RSS_LIMIT:
        raise ValueError('Frozen matrix size or RSS allowance differs')
    rows = []
    for case in cases:
        path = plan/'configs'/(case+'.json')
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.name]:
            raise ValueError('Configuration hash mismatch: '+case)
        cfg = read(path)
        if cfg['case_id'] != case or cfg['engine'] not in ('gradscf', 'pyscf') or cfg['threads'] != 64:
            raise ValueError('Invalid case identity, engine or CPU request: '+case)
        rows.append((case, cfg, path, hashes[path.name]))
    return rows


def previous_attempt(root, case, cfg):
    output = root/'results/matrix-v1'
    paths = [output/case, output/(case+'.process.json'), output/(case+'.log')]
    if cfg['method'] == 'RKS' and cfg['xc'] == 'pbe':
        tag = 'candidate-grid-cache-v3' if cfg['engine'] == 'gradscf' else 'reference-certified-v1'
        paths.append(root/'results'/tag/(case+'.process.json') if cfg['representation'] == 'df'
                     else root/'results/no-ri-pair'/(tag+'-'+cfg['engine']+'.process.json'))
    return next((path for path in paths if path.exists()), None)


def lease_seconds(job):
    end = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%e'], text=True).strip()
    return -1 if not end else datetime.datetime.fromisoformat(end).timestamp()-time.time()


def wait_predecessor(root, pid):
    while pid:
        proc = Path('/proc')/str(pid)
        try:
            command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if not command and (proc/'stat').read_text().split(') ', 1)[1].startswith('Z'):
                return
            if ('continue-no-ri-pair-v2.sh' not in command or
                    (str(root) not in command and not any(os.path.samefile(path, proc/'cwd')
                        for path in (root, root/'harness-reference-v1')))):
                raise RuntimeError('Refusing to wait on an unrelated PID: '+str(pid))
        except FileNotFoundError:
            return
        time.sleep(15)


def wait_idle(job):
    while True:
        lines = subprocess.check_output(['squeue', '--steps', '-h', '-j', job, '-o', '%i|%j'], text=True)
        if not any(line.split('|', 1)[0].strip().split('.')[-1] not in ('batch', 'extern')
                   for line in lines.splitlines() if line.strip()):
            return
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--wait-pid', type=int, default=1884391)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.expanduser().absolute()  # Preserve logical homefile/PWD; never resolve symlinks.
    rows = configurations(root)
    harness = root/'harness-reference-v1'
    worker, guard = (harness/'benchmarks/probes/scf_100/worker.py',
                     harness/'benchmarks/probes/response_grid3/guard.py')
    python = root.parent/'gradscf-libcint-nnao-20261026/env-compare/bin/python'
    if not all(path.is_file() for path in (worker, guard, python)):
        raise FileNotFoundError('Missing frozen harness or existing comparison Python')
    if not all((root/name/'src').is_dir() for name in ('baseline', 'candidate-1a575d18')):
        raise FileNotFoundError('Missing baseline/candidate source')
    if args.dry_run:
        for case, cfg, _, _ in rows:
            prior = previous_attempt(root, case, cfg)
            print(json.dumps(dict(case_id=case, status='skip_previous_attempt' if prior else 'would_run',
                prior_record=str(prior) if prior else None, cap_seconds=cfg['process_limit_seconds'])))
        return
    output = root/'results/matrix-v1'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(root)
    os.environ['PWD'] = str(root)
    def journal(case, status, **fields):
        row = dict(time=datetime.datetime.now().astimezone().isoformat(), job=args.job,
                   case_id=case, status=status, **fields)
        with (output/'journal.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    wait_predecessor(root, args.wait_pid)
    for case, cfg, config, expected_sha in rows:
        prior = previous_attempt(root, case, cfg)
        if prior:
            journal(case, 'skip_previous_attempt', prior_record=str(prior))
            continue
        wait_idle(args.job)  # Conservatively require no other non-batch step, not merely no matching name.
        remaining, cap = lease_seconds(args.job), cfg['process_limit_seconds']
        if remaining <= cap+180:
            journal(case, 'not_run_resource_boundary', remaining_seconds=remaining, required_seconds=cap+180)
            continue
        source = root/('baseline' if cfg['engine'] == 'pyscf' else 'candidate-1a575d18')
        process, log = output/(case+'.process.json'), output/(case+'.log')
        if hashlib.sha256(config.read_bytes()).hexdigest() != expected_sha:
            raise ValueError('Configuration changed after validation: '+case)
        env = dict(os.environ, PWD=str(harness), PYTHONPATH=str(source/'src')+':'+str(harness),
                   GRADSCF_SOURCE_SNAPSHOT=source.name, PYTHONUNBUFFERED='1')
        command = ['srun', '--jobid='+args.job, '--immediate=60', '--overlap', '--gres=none', '--mem=240G',
            '-c64', '-N1', '-n1', '--job-name=scf100-'+case, '--chdir='+str(harness), str(python),
            str(guard), '--seconds', str(cap), '--rss-bytes', str(RSS_LIMIT), str(process), '--',
            str(python), str(worker), str(config), str(output/case)]
        if cfg['engine'] == 'pyscf': command.append('--reference-continuation')
        journal(case, 'launch', source=str(source), input_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
        with log.open('x') as stream:
            code = subprocess.run(command, cwd=str(harness), env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        if not process.exists():
            journal(case, 'unconfirmed_launch', returncode=code)
            break  # Preserve an uncertain attempt and the allocation; never automatically retry.
        result = read(process)
        summary = output/case/'summary.json'
        status = read(summary).get('status', 'missing_scientific_status') if summary.exists() else 'missing_summary'
        if result['status'] != 'ok' or code: status = result['status'] if result['status'] != 'ok' else 'process_error'
        journal(case, status, process_status=result['status'], returncode=code, process_record=str(process))


if __name__ == '__main__':
    main()
