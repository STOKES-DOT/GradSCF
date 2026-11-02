"""Resume the TZP lane on the validated source with the bounded Ne6 repair."""
import argparse
import json
import os
from pathlib import Path

import jax

from gradscf.model.nnao.runtime import atomic_json, file_hash, scientific_source_hash
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool


ROOT = Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021')
HERE = ROOT/'repairs/ne6-tzp-20261023'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=('nnao_tzp_s','nnao_tzp_m','nnao_tzp_l'),required=True)
    family=parser.parse_args().family
    deployment=json.loads((HERE/'deployment.json').read_text())
    gate=json.loads((HERE/'validation.json').read_text())
    assert gate['status']=='passed' and gate['source_sha256']==deployment['new_source_sha256']
    assert scientific_source_hash(HERE/'source')==deployment['new_source_sha256']
    original=json.loads((ROOT/'campaign.json').read_text())
    for name,expected in (original['input_sha256']|original['workflow_sha256']).items():
        assert file_hash(ROOT/name)==expected,name
    assert '--xla_gpu_enable_command_buffer=' in os.environ.get('XLA_FLAGS','')
    if jax.default_backend()!='gpu' or len(jax.devices('gpu'))!=1:
        raise RuntimeError('Require exactly one Slurm-assigned GPU.')
    data=dict(json.loads((ROOT/'inputs/references.json').read_text()),basis_family=family)
    model=dict(json.loads((ROOT/'inputs/model-config.json').read_text()),basis_family=family)
    tier=family.rsplit('_',1)[1]
    candidate=json.loads((ROOT/f'inputs/{tier}-candidate.json').read_text())
    ready=json.loads((ROOT/'ready.json').read_text())
    assert ready['status']=='complete' and ready['candidate_sha256'][tier]==file_hash(ROOT/f'inputs/{tier}-candidate.json')
    output=ROOT/'runs'/family/'train'
    job=os.environ['SLURM_JOB_ID'];attempt=os.environ.get('SLURM_RESTART_COUNT','0')
    record=dict(family=family,job_id=job,restart_count=int(attempt),status='running',
                source_sha256=deployment['new_source_sha256'],finite_differences=False,extra_evaluations=False,
                device_kinds=[d.device_kind for d in jax.devices('gpu')])
    log=HERE/'execution'/f'{family}-{job}-r{attempt}.json'
    atomic_json(log,record)
    try:
        result=run_fixed_pool(data,candidate,ROOT/'cache'/tier,output,
            epochs=1000,learning_rate=1e-3,seed=0,storage='mmap',
            resume=(output/'last.npz').exists(),model_config=model,batch_size=50,
            decay_epochs=100,check_parameter_gradients=False,
            early_stop_window=10,early_stop_tolerance=1e-4)
        assert result['status'] in {'complete','early_stopped'}
        atomic_json(output/'result.json',dict(status=result['status'],epoch=result['epoch'],
            update_step=result['update_step'],loss_hartree=result['loss_hartree'],
            loss_kind=result['loss_kind'],parameters_evaluated=result['parameters_evaluated']))
        record['status']=result['status']
    except Exception as error:
        record.update(status='failed',error=str(error));raise
    finally:atomic_json(log,record)


if __name__=='__main__':main()
