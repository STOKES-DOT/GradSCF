"""Restart only L-TZP after CPU and one-time A100 exact-case validation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import jax

from gradscf.model.nnao.runtime import atomic_json, file_hash, scientific_source_hash
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool


ROOT=Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021')
HERE=ROOT/'repairs/c6cl6-20261026'


def main():
    deployment=json.loads((HERE/'deployment.json').read_text())
    expected=deployment['new_source_sha256']
    gate=json.loads((HERE/'validation.json').read_text())
    assert gate['status']=='passed' and gate['source_sha256']==expected
    assert scientific_source_hash(HERE/'source')==expected
    original=json.loads((ROOT/'campaign.json').read_text())
    for name,digest in (original['input_sha256']|original['workflow_sha256']).items():
        assert file_hash(ROOT/name)==digest,name
    assert '--xla_gpu_enable_command_buffer=' in os.environ.get('XLA_FLAGS','')
    devices=jax.devices('gpu')
    assert len(devices)==1 and 'A100' in devices[0].device_kind and '80GB' in devices[0].device_kind
    # A short exact-case gate runs once per patched source, not per epoch or
    # candidate update. A separate process releases its compiled graphs/memory.
    gpu_gate=HERE/'a100-validation.json'
    if not gpu_gate.exists():
        subprocess.run([sys.executable,'-u',str(HERE/'validate_fix.py'),
                        '--output-name','a100-validation.json'],check=True)
    gpu=json.loads(gpu_gate.read_text())
    assert gpu['status']=='passed' and gpu['source_sha256']==expected
    assert any('A100' in name and '80GB' in name for name in gpu['device_kinds'])
    family='nnao_tzp_l'
    data=dict(json.loads((ROOT/'inputs/references.json').read_text()),basis_family=family)
    model=dict(json.loads((ROOT/'inputs/model-config.json').read_text()),basis_family=family)
    candidate=json.loads((ROOT/'inputs/l-candidate.json').read_text())
    ready=json.loads((ROOT/'ready.json').read_text())
    assert ready['status']=='complete' and ready['candidate_sha256']['l']==file_hash(ROOT/'inputs/l-candidate.json')
    output=ROOT/'runs'/family/'train'
    job=os.environ['SLURM_JOB_ID'];attempt=int(os.environ.get('SLURM_RESTART_COUNT','0'))
    record=dict(family=family,job_id=job,restart_count=attempt,status='running',
        source_sha256=expected,residual_metric='orthonormal_density',seed=0,
        finite_differences=False,extra_training_evaluations=False,
        one_time_exact_case_gate=str(gpu_gate),device_kinds=[d.device_kind for d in devices])
    log=HERE/'execution'/f'{family}-{job}-r{attempt}.json'
    atomic_json(log,record)
    try:
        result=run_fixed_pool(data,candidate,ROOT/'cache/l',output,
            epochs=1000,learning_rate=1e-3,seed=0,storage='mmap',
            resume=(output/'last.npz').exists(),model_config=model,batch_size=50,
            decay_epochs=100,check_parameter_gradients=False,
            early_stop_window=10,early_stop_tolerance=1e-4)
        assert result['status'] in {'complete','early_stopped'}
        atomic_json(output/'result.json',{key:result[key] for key in (
            'status','epoch','update_step','loss_hartree','loss_kind','parameters_evaluated')})
        record['status']=result['status']
    except Exception as error:
        record.update(status='failed',error=str(error));raise
    finally:atomic_json(log,record)


if __name__=='__main__':main()
