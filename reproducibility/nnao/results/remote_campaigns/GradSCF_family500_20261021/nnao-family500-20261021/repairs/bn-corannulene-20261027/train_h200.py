"""Resume the unchanged L-TZP checkpoint on one H200 after its largest-case gate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.model.nnao.runtime import atomic_json,file_hash,scientific_source_hash
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
from gradscf.model.nnao.training import trainer


ROOT=Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021')
HERE=ROOT/'repairs/bn-corannulene-20261027'
SOURCE=ROOT/'repairs/c6cl6-20261026/source'
EXPECTED='c28a091033269e2a50f935050ab481e2480fa5566ef2a45099a6e5a47703179c'


def main():
    assert scientific_source_hash(SOURCE)==EXPECTED
    devices=jax.devices('gpu')
    assert len(devices)==1 and 'H200' in devices[0].device_kind
    assert jax.config.jax_enable_x64 and '--xla_gpu_enable_command_buffer=' in os.environ.get('XLA_FLAGS','')
    original=json.loads((ROOT/'campaign.json').read_text())
    for name,expected in (original['input_sha256']|original['workflow_sha256']).items():
        assert file_hash(ROOT/name)==expected,name
    output=ROOT/'runs/nnao_tzp_l/train'
    checkpoint=output/'last.npz'
    with np.load(checkpoint,allow_pickle=False) as saved:
        metadata=json.loads(saved['metadata'].item())
    identity=metadata['identity']
    assert identity['scientific_source_sha256']==EXPECTED and identity['backend']=='gpu'
    assert identity['jax']==jax.__version__ and metadata['update_step']>=3
    job=os.environ['SLURM_JOB_ID'];restart=int(os.environ.get('SLURM_RESTART_COUNT','0'))
    record=dict(status='running',phase='h200_exact_case_gate',job_id=job,restart_count=restart,
        family='nnao_tzp_l',device_kind=devices[0].device_kind,source_sha256=EXPECTED,
        resume_update=metadata['update_step'],optimizer_reset=False,finite_differences=False,
        extra_epoch_evaluations=False)
    path=HERE/'execution'/f'nnao_tzp_l-{job}-r{restart}.json'
    atomic_json(path,record)
    real_factory=trainer.make_mace_predictor
    try:
        gate=HERE/'probe-h200.json'
        if not gate.exists() or json.loads(gate.read_text()).get('status')!='passed':
            subprocess.run([sys.executable,'-u',str(HERE/'probe.py'),
                            '--output-name','probe-h200.json'],check=True)
        checked=json.loads(gate.read_text())
        assert checked['status']=='passed' and 'H200' in checked['device']
        assert checked['source_sha256']==EXPECTED and checked['full_mace_backward_finite']
        # Resume identity includes the seed-0 initialization hash. Restore its
        # recorded bytes for identity construction, avoiding harmless GPU-arch
        # rounding changes. Training still loads theta_3 and its saved Adam.
        initial_path=ROOT/'repairs/c6cl6-20261026/before-restart/failed-update-000001.npz'
        with np.load(initial_path,allow_pickle=False) as saved:
            initial=np.array(saved['parameters'])
        assert hashlib.sha256(initial.tobytes()).hexdigest()==identity['initialization_sha256']
        def recorded_initialization(seed,model_config,*,return_mask=False):
            assert seed==identity['seed'] and model_config==identity['model_config']
            result=real_factory(seed,model_config,return_mask=return_mask)
            fresh=np.asarray(result[0])
            np.testing.assert_allclose(fresh,initial,atol=5e-14,rtol=5e-14)
            if return_mask:
                assert np.flatnonzero(~result[3]).tolist()==identity['frozen_parameter_indices']
            record['seed_initialization_max_abs_roundoff']=float(np.max(np.abs(fresh-initial)))
            atomic_json(path,record)
            return (jnp.asarray(initial),)+result[1:]
        trainer.make_mace_predictor=recorded_initialization
        data=dict(json.loads((ROOT/'inputs/references.json').read_text()),basis_family='nnao_tzp_l')
        model=dict(json.loads((ROOT/'inputs/model-config.json').read_text()),basis_family='nnao_tzp_l')
        candidate=json.loads((ROOT/'inputs/l-candidate.json').read_text())
        ready=json.loads((ROOT/'ready.json').read_text())
        assert ready['status']=='complete' and ready['candidate_sha256']['l']==file_hash(ROOT/'inputs/l-candidate.json')
        record['phase']='resume_training';atomic_json(path,record)
        result=run_fixed_pool(data,candidate,ROOT/'cache/l',output,epochs=1000,
            learning_rate=1e-3,seed=0,storage='mmap',resume=True,model_config=model,
            batch_size=50,decay_epochs=100,check_parameter_gradients=False,
            early_stop_window=10,early_stop_tolerance=1e-4)
        assert result['status'] in {'complete','early_stopped'}
        atomic_json(output/'result.json',{key:result[key] for key in (
            'status','epoch','update_step','loss_hartree','loss_kind','parameters_evaluated')})
        record.update(status=result['status'],phase='complete')
    except Exception as error:
        record.update(status='failed',error=str(error));raise
    finally:
        trainer.make_mace_predictor=real_factory
        atomic_json(path,record)


if __name__=='__main__':main()
