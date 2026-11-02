"""Submit eight jobs in three serial zeta lanes, after one preparation job."""
from pathlib import Path
import json
import subprocess
import time

from gradscf.model.nnao.runtime import atomic_json, file_hash, repository_root, scientific_source_hash

ROOT=Path(__file__).resolve().parent
project=repository_root()
definition=json.loads((ROOT/'campaign.json').read_text())
if scientific_source_hash(project)!=definition['scientific_source_sha256']:
    raise ValueError('Uploaded scientific source differs from the reviewed campaign.')
for name,expected in (definition['input_sha256']|definition['workflow_sha256']).items():
    if file_hash(ROOT/name)!=expected:
        raise ValueError('Uploaded experiment input differs: '+name)
record=ROOT/'submitted.json'
state=json.loads(record.read_text()) if record.exists() else dict(
    status='submitting',max_simultaneous_gpus=3,prepare=None,lanes={})
if state['status']=='submitted':
    raise ValueError('All jobs are already submitted; no duplicates will be created.')


def submit(name,dependency=None):
    command=['sbatch','--parsable']
    if dependency:command.append('--dependency='+dependency)
    script=ROOT/(name+'.slurm')
    command.append(str(script))
    for attempt in range(4):
        time.sleep(2.2)
        result=subprocess.run(command,cwd=project,text=True,capture_output=True)
        if result.returncode and 'Submission too frequent' in result.stderr and attempt<3:
            continue
        if result.returncode:
            raise RuntimeError(result.stderr)
        output=result.stdout
        break
    job_id=int(output.strip().splitlines()[-1].split(';')[0])
    print(json.dumps(dict(name=name,job_id=job_id,dependency=dependency)),flush=True)
    return dict(job_id=job_id,dependency=dependency,script_sha256=file_hash(script))


if state['prepare'] is None:
    state['prepare']=submit('prepare')
atomic_json(record,state)
for zeta,families in definition['lanes'].items():
    existing=state['lanes'].setdefault(zeta,[])
    if [r['family'] for r in existing]!=families[:len(existing)]:
        raise ValueError('Previously submitted lane differs from this campaign.')
    dependency='afterok:'+str(state['prepare']['job_id'])
    for family in families:
        previous=next((r for r in existing if r['family']==family),None)
        if previous is not None:
            dependency='afterany:'+str(previous['job_id'])
            continue
        job=submit(family,dependency)
        state['lanes'][zeta].append(dict(family=family,**job))
        atomic_json(record,state)
        dependency='afterany:'+str(job['job_id'])
state['status']='submitted'
atomic_json(record,state)
