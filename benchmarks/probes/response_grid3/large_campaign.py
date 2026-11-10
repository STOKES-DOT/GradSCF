"""Serial plan execution through an existing allocation, with prerequisite gates."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('campaign', type=Path)
parser.add_argument('allocation')
args = parser.parse_args()
root = args.campaign.resolve()
source = root/'candidate-final'
python = root.parent/'gradscf-libcint-nnao-20261026/env-compare/bin/python'
plan = json.loads((source/'reports/response_grid3/20261101-2043d7d/configs/first_large_true_ri/run_plan.json').read_text())
output = root/('results/scaling-'+args.allocation)
output.mkdir(exist_ok=False)
accepted = {}
try:
    for stage in ('prerequisites', 'largest_serial_order'):
        for item in plan[stage]:
            if not item.get('enabled', True):
                continue
            name = Path(item['config']).stem
            dependency = item.get('required_before')
            if dependency and not accepted.get(dependency, False):
                (output/(name+'.skipped.json')).write_text(json.dumps(dict(status='blocked_prerequisite', dependency=dependency)))
                continue
            end = subprocess.check_output(['squeue', '-h', '-j', args.allocation, '-o', '%e'], text=True).strip()
            remaining = int(datetime.datetime.fromisoformat(end).timestamp()-time.time())-120
            limit = min(6000 if stage == 'largest_serial_order' else 600, remaining)
            if limit < 300:
                raise RuntimeError('Insufficient remaining allocation time; unstarted cases stay unaccepted.')
            threads = 64 if stage == 'largest_serial_order' else 8
            command = [x.replace('<python>', str(python)).replace('<output_root>', str(output)) for x in item['argv']]
            env = dict(os.environ, PYTHONPATH=str(source/'src')+':'+str(source), JAX_PLATFORMS='cpu',
                JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', GRADSCF_NATIVE_THREADS=str(threads),
                GRADSCF_SOURCE_COMMIT='2b18b85', PYTHONUNBUFFERED='1')
            launch = ['srun', '--jobid='+args.allocation, '--exact', '--gres=none', '--mem=240G',
                '-N1', '-n1', '-c'+str(threads), '--chdir='+str(source), str(python),
                'benchmarks/probes/response_grid3/guard.py', '--seconds', str(limit),
                '--rss-bytes', str(200*1024**3), str(output/(name+'.process.json')), '--', *command]
            with (output/(name+'.log')).open('w') as log:
                code = subprocess.run(launch, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if 'tddft.py' in command[1]:
                status = Path(command[3])/'status.json'
                good = status.exists() and json.loads(status.read_text()).get('scientific_pass', False)
            else:
                path = Path(command[2])
                rows = [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]
                good = any(r.get('stage') == 'workflow' and r.get('scientific_pass') for r in rows)
            accepted[item['config']] = bool(code == 0 and good)
            (output/'acceptance.json').write_text(json.dumps(accepted, indent=2)+'\n')
finally:
    (root/('release.'+args.allocation)).touch()
