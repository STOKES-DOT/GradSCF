"""Replace the blocked TZP chain after the exact Ne6 GPU validation passes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT=Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021')
CAMPAIGN=ROOT/'artifacts/nnao-family500-20261021'
HERE=CAMPAIGN/'repairs/ne6-tzp-20261023'


def main():
    assert not (HERE/'submitted.json').exists(), 'Review an existing repair submission before resubmitting.'
    validation=json.loads((HERE/'validation.json').read_text())
    assert validation['status']=='passed'
    deployment=json.loads((HERE/'deployment.json').read_text())
    assert validation['source_sha256']==deployment['new_source_sha256']
    os.chdir(ROOT);os.environ['PWD']=str(ROOT)
    record=dict(status='submitting',source_sha256=deployment['new_source_sha256'],
                finite_differences=False,max_simultaneous_campaign_gpus=3,jobs=[])
    def save(): (HERE/'submitted.json').write_text(json.dumps(record,indent=2)+'\n')
    save();previous=None
    try:
        for family,step in (('nnao_tzp_s',435),('nnao_tzp_m',0),('nnao_tzp_l',0)):
            script=HERE/(family+'.slurm')
            script.write_text(f'''#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name={family}_ne6fix
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --exclude=n005
#SBATCH --chdir={ROOT}
#SBATCH --output=artifacts/nnao-family500-20261021/repairs/ne6-tzp-20261023/{family}-%j.log
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/artifacts/nnao-family500-20261021/repairs/ne6-tzp-20261023/source/src:$PWD/artifacts/nnao-family500-20261021/repairs/ne6-tzp-20261023/source/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${{XLA_FLAGS:-}} --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-family500-20261021/jit-cache/ne6fix/{family}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
date -Is
requeue_for_time_limit() {{
    trap - USR1
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}}
trap requeue_for_time_limit USR1
python -u artifacts/nnao-family500-20261021/repairs/ne6-tzp-20261023/train.py --family {family} &
worker=$!
wait "$worker"
trap - USR1
date -Is
''')
            subprocess.run(['bash','-n',str(script)],check=True)
            args=['sbatch','--parsable']
            parent=previous if previous is not None else 104593
            args.append('--dependency=afterok:'+str(parent))
            args.append(str(script))
            result=subprocess.run(args,text=True,capture_output=True)
            if result.returncode:raise RuntimeError(result.stdout+result.stderr)
            job=int(result.stdout.strip().splitlines()[-1].split(';')[0])
            row=dict(family=family,job_id=job,seed_update=step,
                     dependency='afterok:'+str(parent),
                     script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                     resources=dict(gpu='A100_80G',memory_gib=512,time_limit_hours=72),
                     source_sha256=deployment['new_source_sha256'])
            record['jobs'].append(row);save();previous=job
            print(json.dumps(row),flush=True);time.sleep(2.5)
        # These queued jobs still point to the failed, old-source parent.
        for job,name in ((104097,'nnao_tzp_m_goff'),(104099,'nnao_tzp_l_goff')):
            info=subprocess.check_output(['scontrol','show','job','-o',str(job)],text=True)
            assert 'JobName='+name+' ' in info and 'JobState=PENDING ' in info,info
            subprocess.run(['scancel',str(job)],check=True)
        record['replaced_pending_jobs']=[104097,104099]
        index=CAMPAIGN/'restarts/graph-off-20261023/submitted.json'
        old=json.loads(index.read_text())
        archive=index.with_name('submitted-before-ne6-tzp-fix.json')
        assert not archive.exists();archive.write_text(index.read_text())
        old['lanes']['tzp']=record['jobs'];old['ne6_tzp_repair']=str(HERE)
        index.write_text(json.dumps(old,indent=2)+'\n')
        record['status']='submitted'
    except Exception as error:
        record.update(status='submission_failed',error=str(error));raise
    finally:save()


if __name__=='__main__':main()
