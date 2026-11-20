"""Local scheduler client: serial fresh-process steps in one existing allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shlex
import signal
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='jy_gpu_2')
    parser.add_argument('--root', required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-directory', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    remote_root = args.root.rstrip('/')
    python = '/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin/python'
    options = ['-o','BatchMode=yes','-o','ConnectTimeout=12',
               '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2']
    def remote(command, *, timeout=90, log=None):
        return subprocess.run(['ssh',*options,args.host,command],text=True,
            stdout=log if log else subprocess.PIPE, stderr=log if log else subprocess.PIPE,
            timeout=timeout, check=True)
    def step(command, *, gpu):
        environment = ['env',f'PYTHONPATH={remote_root}/source/src',
            'JAX_PLATFORMS=cuda' if gpu else 'JAX_PLATFORMS=cpu', 'JAX_ENABLE_X64=1',
            'XLA_PYTHON_CLIENT_PREALLOCATE=false','XLA_PYTHON_CLIENT_MEM_FRACTION=0.80',
            'OMP_NUM_THREADS=8','OPENBLAS_NUM_THREADS=8','MKL_NUM_THREADS=8',
            f'MPLCONFIGDIR={remote_root}/results/mpl']
        arguments = ['srun',f'--jobid={args.job_id}','--exclusive','--ntasks=1',
                     '--cpus-per-task=32','--cpu-bind=threads',*environment,python,'-u',*command]
        return 'cd '+shlex.quote(remote_root+'/source')+' && '+shlex.join(arguments)
    def write(path, value):
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    source = Path(__file__).resolve().parent
    expected = {key:hashlib.sha256((source/name).read_bytes()).hexdigest() for key,name in
                [('benchmark_source','grid_benchmark.py'),('matrix_source','molecule.py')]}
    ids, blocks = ['010','017','065'], [256,512,1024,2048,4096,8192,16384]
    randomizer = random.Random(20261117)
    failures, commands, completed = [], [], []
    def stop(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try:
        for identifier in ids:
            tail=blocks[1:].copy();randomizer.shuffle(tail)
            for block in [256,*tail]:
                key=f'{identifier}-{block}'
                write(args.output/'progress.json',{'phase':'running','current':key,
                    'completed':len(completed),'total':21,'job_id':args.job_id})
                path=f'{remote_root}/results/{identifier}/block-{block}.json'
                existing=remote('if test -f '+shlex.quote(path)+'; then cat '+shlex.quote(path)+'; fi').stdout
                record=json.loads(existing) if existing.strip() else None
                if record and any(record.get(k)!=v for k,v in expected.items()):
                    raise RuntimeError('Existing benchmark source differs; use a new results directory')
                command=step(['tools/xc100_grid_point.py','--root',remote_root,
                              '--id',identifier,'--block-size',str(block),'--repeats','3'],gpu=True)
                commands.append({'system_id':identifier,'block_size':block,
                                 'command':command,'reused_existing_point':record is not None})
                if record is None:
                    print('RUN',key,flush=True)
                    start=time.monotonic()
                    try:
                        with (args.output/f'{key}.log').open('w') as log:
                            remote(command,timeout=600,log=log)
                    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                        failures.append({'system_id':identifier,'block_size':block,
                            'error':str(exc),'elapsed_seconds':time.monotonic()-start})
                    saved=remote('if test -f '+shlex.quote(path)+'; then cat '+shlex.quote(path)+'; fi').stdout
                    record=json.loads(saved) if saved.strip() else None
                if record:
                    write(args.output/f'{key}.json',record)
                completed.append(key)
                print('DONE',key,record.get('status') if record else 'failed',flush=True)
        sweep={'ids':ids,'blocks':blocks,'repeats':3,'order_seed':20261117,
               'completed':completed,'failures':failures,'commands':commands,
               'job_id':args.job_id,'host':args.host,'source_hashes':expected}
        write(args.output/'sweep.json',sweep)
        with tempfile.TemporaryDirectory(prefix='gradscf-grid-sweep-') as directory:
            path=Path(directory)/'sweep.json';write(path,sweep)
            subprocess.run(['scp',*options,str(path),f'{args.host}:{remote_root}/results/sweep.json'],check=True,timeout=90)
        remote(step(['tools/xc100_grid_report.py','--root',remote_root],gpu=False),timeout=120)
        args.report_directory.mkdir(parents=True,exist_ok=True)
        names=['summary.json','summary.csv','grid_block_scaling.png','grid_block_scaling.pdf','REPORT.md']
        subprocess.run(['scp',*options,*[f'{args.host}:{remote_root}/results/{name}' for name in names],
                        str(args.report_directory)],check=True,timeout=120)
        write(args.output/'progress.json',{'phase':'complete','completed':len(completed),
              'total':21,'failures':len(failures),'report_directory':str(args.report_directory)})
    finally:
        try:remote('touch '+shlex.quote(remote_root+'/release.flag'))
        except Exception as exc:write(args.output/'release_error.json',{'error':str(exc)})


if __name__=='__main__':
    main()
