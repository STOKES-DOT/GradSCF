"""Submit an explicit, idempotently recorded three-lane Slurm dependency graph."""
import json
from pathlib import Path
import subprocess
import time

root = Path('artifacts/basis-grid-20261013')
plan = json.loads((root/'job-plan.json').read_text())
record = root/'submitted-jobs.json'
submitted = json.loads(record.read_text()) if record.exists() else []
by_family = {r['family']: r for r in submitted}
last = {'szp': plan['existing_szp_l_job']}
for job in plan['jobs']:
    family, lane = job['family'], job['lane']
    if family in by_family:
        last[lane] = by_family[family]['job_id']
        continue
    dependencies = sorted({plan['validation_job'], *([last[lane]] if lane in last else [])})
    dep = 'afterok:'+':'.join(map(str, dependencies))
    result = subprocess.run(['sbatch', '--parsable', f'--dependency={dep}',
                             '--kill-on-invalid-dep=yes', job['script']],
                            text=True, capture_output=True, check=True)
    job_id = int(result.stdout.strip().splitlines()[-1].split(';')[0])
    row = dict(job, job_id=job_id, dependencies=dependencies)
    submitted.append(row)
    temporary = record.with_suffix('.tmp')
    temporary.write_text(json.dumps(submitted, indent=2)+'\n')
    temporary.replace(record)
    last[lane] = job_id
    print(json.dumps(row), flush=True)
    time.sleep(2)
