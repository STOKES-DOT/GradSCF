"""Replay the exact ADIM6/AD6 epoch-444 candidate before restarting epoch 443."""
import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.data.molecule import atomic_number
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json,cache_location,file_hash
from gradscf.model.nnao.training.trainer import make_mace_predictor

OUT=Path(__file__).resolve().parent
PARENT=Path('/home/user_yuanjiao/private/homefile/GradSCF')
training=PARENT/'artifacts/nnao-fd-precision-20261013/training'
mode=sys.argv[1]
assert mode in ('baseline','fixed')
if mode=='baseline':
    name='gradscf.scf._stability_parent'
    spec=importlib.util.spec_from_file_location(name,OUT/'stability-parent.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    import gradscf.scf.stability as stability
    stability.stabilize_df_rhf=module.stabilize_df_rhf

with np.load(training/'last.npz',allow_pickle=False) as saved:
    checkpoint=json.loads(str(saved['metadata'].item()))
identity=checkpoint['identity'];assert checkpoint['epoch']==443
with np.load(training/'failed-candidate-epoch-000444.npz',allow_pickle=False) as saved:
    vector=jnp.asarray(saved['parameters'])
row=next(r for r in identity['references']['structures'] if r['structure_id']=='ADIM6/AD6')
with jax.default_device(jax.devices('cpu')[0]):
    layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
        basis_family=identity['references']['basis_family'],core_primitives=identity['references']['core_primitives'])
    layout=bind_pool_candidate(layout,identity['candidate'])
problem,_=prepare_fixed_pool(row,layout,cache_location(PARENT/'artifacts/fixed-pool200-20261012/cache',row),storage='mmap')
_,predict,backward,mask=make_mace_predictor(identity['seed'],identity['model_config'],return_mask=True)
graph=build_graph([atomic_number(s) for s in row['symbols']],row['coords_angstrom'],
                   element_order=identity['model_config']['elements'])

def evaluate(parameters):
    coefficients=predict(parameters,graph)
    (energy,info),derivative=problem._value_grad(coefficients,*problem.raw)
    gradient=backward(parameters,graph,derivative)
    return float(energy),np.asarray(gradient),{k:np.asarray(v).item() for k,v in info.items()}

energy,gradient,info=evaluate(vector)
record=dict(mode=mode,structure_id=row['structure_id'],energy_hartree=energy,
    finite_gradient=bool(np.isfinite(gradient).all()),scf=info,backend=jax.default_backend(),
    candidate_sha256=file_hash(training/'failed-candidate-epoch-000444.npz'))
if mode=='fixed':
    assert info['converged'] and info['stable'] and info['stability_resolved'],info
    assert record['finite_gradient']
    direction=np.random.default_rng(0).normal(size=vector.shape)*mask
    direction/=np.linalg.norm(direction);step=1e-4
    plus=evaluate(vector+step*direction)
    minus=evaluate(vector-step*direction)
    for result in (plus,minus):
        assert result[2]['stable'] and np.isfinite(result[1]).all()
    ad=float(gradient@direction);fd=(plus[0]-minus[0])/(2*step)
    record['directional_check']=dict(ad=ad,fd=fd,step=step,atol=2e-6,rtol=2e-5,
                                    passed=bool(np.isclose(ad,fd,atol=2e-6,rtol=2e-5)))
    assert record['directional_check']['passed'],record['directional_check']
atomic_json(OUT/f'ad6-{mode}.json',record)
print(json.dumps(record),flush=True)
