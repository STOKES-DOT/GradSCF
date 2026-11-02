"""Approved 256-channel MACE, minibatch-50 fixed-pool training."""
import argparse
import json
import os
from pathlib import Path
import jax
from gradscf.model.nnao.runtime import scientific_source_hash,repository_root,atomic_json
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool

ROOT=Path(__file__).resolve().parent

def main(mode):
    source=json.loads((ROOT/'source.json').read_text())
    assert scientific_source_hash(repository_root())==source['source_sha256'],'Source mismatch'
    if source.get('parent_source_sha256'):
        verification=json.loads((ROOT/'atomic-rescue-verification.json').read_text())
        assert verification.get('passed') and verification['source_sha256']==source['source_sha256']
    assert jax.config.jax_enable_x64 and jax.default_backend()=='gpu'
    candidate=json.loads((ROOT/'selection/candidate.json').read_text())
    model=json.loads((ROOT/'inputs/model-config.json').read_text())
    filename='smoke-references.json' if mode=='smoke' else 'references.json'
    data=json.loads((ROOT/'inputs'/filename).read_text())
    output=ROOT/mode;resume=(output/'last.npz').exists()
    if mode=='train':
        gate=json.loads((ROOT/'smoke-passed.json').read_text())
        assert gate['passed'] and gate['source_sha256']==source['source_sha256']
    result=run_fixed_pool(data,candidate,ROOT/'cache',output,epochs=1 if mode=='smoke' else 1000,
        learning_rate=1e-3,seed=0,storage='mmap',resume=resume,model_config=model,
        batch_size=50,decay_epochs=100)
    if mode=='smoke':
        assert result['epoch']==1 and result['update_step']==1
        atomic_json(ROOT/'smoke-passed.json',dict(passed=True,source_sha256=source['source_sha256'],
            structures=len(data['structures']),element_count=len(data['elements']),epoch=1,
            parameter_count=len(result['parameters']),dtype=str(result['parameters'].dtype),
            slurm_job_id=os.environ.get('SLURM_JOB_ID')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train']);a=p.parse_args();main(a.mode)
