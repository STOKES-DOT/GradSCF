"""Strict epoch-443 migration and 200-molecule replay with Adam preserved."""
import importlib.util
import json
from pathlib import Path
import sys

import jax
import numpy as np
from gradscf.model.nnao.runtime import atomic_json,file_hash,scientific_source_hash
from gradscf.model.nnao.training import trainer
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PARENT=Path('/home/user_yuanjiao/private/homefile/GradSCF')
OLD=PARENT/'artifacts/nnao-fd-precision-20261013'
EXPECTED='ec3b63f54f67fd89069bcd1a6992dc8732d5bb3380a4ea49b71818d0e39c7ff6'
spec=importlib.util.spec_from_file_location('accepted_migration',OLD/'resume.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

assert scientific_source_hash(PARENT)==EXPECTED,'Parent source changed'
def sources(root):
    package=root/'src/gradscf'
    return {str(p.relative_to(package)):file_hash(p) for p in package.rglob('*')
            if p.is_file() and p.suffix in {'.py','.json','.dat'}}
before,after=sources(PARENT),sources(ROOT)
changed=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
assert changed==['scf/stability.py'],changed
gate=json.loads((OUT/'ad6-fixed.json').read_text())
assert gate['finite_gradient'] and gate['scf']['stable'] and gate['directional_check']['passed']
parent,arrays=helper.read_checkpoint(OLD/'training/last.npz')
assert parent['epoch']==443
identity=dict(parent['identity']);identity['scientific_source_sha256']=scientific_source_hash(ROOT)
records={}
for name in ('last.npz','best.npz'):
    target=OUT/'training'/name
    if not target.exists():
        records[name]=helper.migrate_checkpoint(OLD/'training'/name,target,identity,parent_source=EXPECTED)
atomic_json(OUT/'migration.json',dict(parent_epoch=443,source_changes=changed,
    target_source=identity['scientific_source_sha256'],optimizer_reset=False,checkpoints=records))
current,state_arrays=helper.read_checkpoint(OUT/'training/last.npz')
assert current['epoch']==443
for name in arrays:assert arrays[name].tobytes()==state_arrays[name].tobytes()
mask=np.ones(arrays['parameters'].shape,dtype=bool)
mask[identity['frozen_parameter_indices']]=False
optimizer=trainer.Adam(identity['learning_rate'],trainable_mask=mask)
state=optimizer.decode(arrays['optimizer'].tobytes(),optimizer.init(arrays['parameters']))
assert optimizer.encode(state)==arrays['optimizer'].tobytes()
counts=[int(x) for x in jax.tree.leaves(state) if np.ndim(x)==0 and np.asarray(x).dtype.kind in 'iu']
assert counts and all(c==443 for c in counts),counts
with np.load(OLD/'training/failed-candidate-epoch-000444.npz',allow_pickle=False) as saved:
    expected_trial=np.array(saved['parameters'])
trial,_=optimizer.update(arrays['parameters'],arrays['gradient'],state)
np.testing.assert_allclose(trial,expected_trial,atol=1e-14,rtol=0)
atomic_json(OUT/'optimizer-check.json',dict(epoch=443,counts=counts,bytes_preserved=True,
    next_candidate_matches=True,parameters_preserved=True,gradient_preserved=True))
original_batch=trainer.evaluate_batch
replays=[]
def guarded_batch(vector,evaluators,**kwargs):
    result=original_batch(vector,evaluators,**kwargs)
    if np.array_equal(vector,arrays['parameters']):
        replays.append(helper.verify_replay(parent,arrays,result))
        atomic_json(OUT/'restart-gates.json',dict(passed=True,replays=replays))
    return result
trainer.evaluate_batch=guarded_batch
result=run_fixed_pool(identity['references'],identity['candidate'],
    PARENT/'artifacts/fixed-pool200-20261012/cache',OUT/'training',epochs=1000,
    learning_rate=identity['learning_rate'],seed=identity['seed'],storage=identity['storage'],resume=True)
sys.exit(int(result['status']!='complete'))
