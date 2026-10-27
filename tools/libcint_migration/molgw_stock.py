"""Run one unchanged upstream MOLGW input as a capability/NaN diagnostic.

This is not the water performance comparison. No GradSCF orbitals, QP files,
comparison executable or numerical-source patch enters the reference run.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('input', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('--intel-runtime', type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    basis = work.parent.parent/'basis'
    if not basis.exists():
        basis.symlink_to(executable.parent/'basis', target_is_directory=True)
    if basis.resolve() != (executable.parent/'basis').resolve():
        raise ValueError('Reference basis directory differs from the upstream build')
    shutil.copy2(args.input, work/'input.in')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    env = {k: v for k, v in os.environ.items() if not k.startswith('MOLGW_COMPARE_')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_STACKSIZE='256M')
    if args.intel_runtime is not None:
        env['LD_LIBRARY_PATH'] = str(args.intel_runtime)+':'+env.get('LD_LIBRARY_PATH', '')
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    limit = 256*1024**2 if hard == resource.RLIM_INFINITY else min(hard, 256*1024**2)
    def stack():
        resource.setrlimit(resource.RLIMIT_STACK, (limit, hard))
    report = dict(status='running', scope='unmodified upstream BSE-scissor capability diagnostic',
        command=[str(executable), 'input.in'], source_input=str(args.input),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        stock_revision='b831818d7a845c36f295d036dc8ceef59daf9991',
        input_unchanged=(args.input.read_bytes() == (work/'input.in').read_bytes()),
        affinity=sorted(os.sched_getaffinity(0)), stack_bytes=limit,
        comparison_environment_variables={k: v for k, v in env.items() if k.startswith('MOLGW_COMPARE_')})
    (work/'ldd.json').write_text(json.dumps(subprocess.run(['ldd', str(executable)], env=env,
        capture_output=True, text=True).__dict__, default=str, indent=2)+'\n')
    tick = time.perf_counter()
    try:
        with (work/'stdout.log').open('w') as log:
            result = subprocess.run(report['command'], cwd=work, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=6000, preexec_fn=stack)
        report['returncode'] = result.returncode
        report['normal_end'] = 'This is the end' in (work/'stdout.log').read_text()
        import yaml
        data = yaml.safe_load((work/'molgw.yaml').read_text())
        excitations = data['optical spectrum']['excitations']
        def values(mapping):
            return [float(mapping[k]) for k in sorted(k for k in mapping if isinstance(k, int))]
        energies, strengths = values(excitations['energies']), values(excitations['oscillator strengths'])
        finite = bool(energies) and all(math.isfinite(x) for x in energies+strengths)
        report.update(energies_ev=energies, oscillator_strengths=strengths, all_finite=finite,
            status='ok' if result.returncode == 0 and report['normal_end'] and finite else 'numerical_mismatch')
    except subprocess.TimeoutExpired:
        report['status'] = 'timeout'
    except Exception as error:
        report.update(status='error', error=repr(error))
    report.update(wall_seconds=time.perf_counter()-tick,
        peak_child_rss_bytes=1024*resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        memory_method='Linux maximum child ru_maxrss; includes ldd helper',
        time_ratio=None, comparability='different upstream small case; no GradSCF speed ratio')
    (work/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], seconds=report['wall_seconds'], work=str(work))))


if __name__ == '__main__':
    main()
