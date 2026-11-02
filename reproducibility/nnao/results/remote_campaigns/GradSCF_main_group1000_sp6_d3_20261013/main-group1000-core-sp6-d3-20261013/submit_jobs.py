"""Submit and record the explicit nine-job graph without touching job 96510."""
import json
from pathlib import Path
import subprocess
import time

root=Path('artifacts/main-group1000-core-sp6-d3-20261013')
plan=json.loads((root/'job-plan.json').read_text())
record=root/'submitted-jobs.json'
submitted=json.loads(record.read_text()) if record.exists() else []
known={r['family']:r for r in submitted}
last={'szp':plan['retained_original_job']}
for job in plan['jobs']:
    family,lane=job['family'],job['lane']
    if family in known:
        last[lane]=known[family]['job_id']
        continue
    deps=sorted({plan['validation_job'],*([last[lane]] if lane in last else [])})
    result=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+':'.join(map(str,deps)),
        job['script']],text=True,capture_output=True,check=True)
    identifier=int(result.stdout.strip().splitlines()[-1].split(';')[0])
    row=dict(job,job_id=identifier,dependencies=deps)
    submitted.append(row)
    temporary=record.with_suffix('.tmp')
    temporary.write_text(json.dumps(submitted,indent=2)+'\n');temporary.replace(record)
    last[lane]=identifier
    print(json.dumps(row),flush=True)
    time.sleep(2)
