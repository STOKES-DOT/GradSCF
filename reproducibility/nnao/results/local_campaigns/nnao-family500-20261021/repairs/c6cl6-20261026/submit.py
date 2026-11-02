"""Submit L-TZP only; preserve the completed families and active L-DZP lane."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT=Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021')
CAMPAIGN=ROOT/'artifacts/nnao-family500-20261021'
HERE=CAMPAIGN/'repairs/c6cl6-20261026'


def main():
    assert not (HERE/'submitted.json').exists(),'Inspect the existing submission before resubmitting.'
    deployment=json.loads((HERE/'deployment.json').read_text())
    gate=json.loads((HERE/'validation.json').read_text())
    assert gate['status']=='passed' and gate['source_sha256']==deployment['new_source_sha256']
    os.chdir(ROOT);os.environ['PWD']=str(ROOT)
    live=subprocess.check_output(['squeue','-u','user_yuanjiao','-h','-o','%i|%j|%T'],text=True)
    assert not any('nnao_tzp_l' in row for row in live.splitlines()),live
    assert len([row for row in live.splitlines() if 'nnao_' in row and '|RUNNING' in row])<3,live
    script=HERE/'nnao_tzp_l.slurm'
    subprocess.run(['bash','-n',str(script)],check=True)
    result=subprocess.run(['sbatch','--parsable',str(script)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    job=int(result.stdout.strip().splitlines()[-1].split(';')[0])
    record=dict(status='submitted',family='nnao_tzp_l',job_id=job,previous_job_id=104598,
        seed_update=0,seed=0,dependency=None,finite_differences=False,
        source_sha256=deployment['new_source_sha256'],
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        resources=dict(gpu='A100_80G',memory_gib=512,cpus=32,time_limit_hours=72),
        max_simultaneous_campaign_gpus=3,residual_metric='orthonormal_density')
    (HERE/'submitted.json').write_text(json.dumps(record,indent=2)+'\n')
    index=CAMPAIGN/'restarts/graph-off-20261023/submitted.json'
    previous=json.loads(index.read_text())
    archive=index.with_name('submitted-before-c6cl6-fix.json');assert not archive.exists()
    archive.write_text(index.read_text())
    matches=[r for r in previous['lanes']['tzp'] if r['family']=='nnao_tzp_l']
    assert len(matches)==1 and matches[0]['job_id']==104598,matches
    previous['lanes']['tzp']=[record if r['family']=='nnao_tzp_l' else r for r in previous['lanes']['tzp']]
    previous['c6cl6_l_tzp_repair']=str(HERE)
    index.write_text(json.dumps(previous,indent=2)+'\n')
    info=subprocess.check_output(['scontrol','show','job','-o',str(job)],text=True)
    assert 'MinMemoryNode=512G' in info and 'NumCPUs=32' in info and 'gpu:A100_80G:1' in info,info
    (HERE/'queue-verification.txt').write_text(info)
    print(json.dumps(record,indent=2),flush=True)
    print(info,flush=True)


if __name__=='__main__':main()
