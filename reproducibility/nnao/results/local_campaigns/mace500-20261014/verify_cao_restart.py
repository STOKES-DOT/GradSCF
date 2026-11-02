"""Gate the actual failed parameters, implicit AD, and checkpoint migration."""
from pathlib import Path
import hashlib
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.runtime import atomic_json,cache_location,scientific_source_hash,repository_root,file_hash
from gradscf.integrals._native import register_integrals

ROOT=Path(__file__).resolve().parent
source=json.loads((ROOT/'source.json').read_text())['source_sha256']
assert scientific_source_hash(repository_root())==source
data=json.loads((ROOT/'inputs/references.json').read_text())
config=json.loads((ROOT/'inputs/model-config.json').read_text())
candidate=json.loads((ROOT/'selection/candidate.json').read_text())
old_records=json.loads((ROOT/'train/epoch-molecules.json').read_text())
assert old_records['epoch']==1
reference={r['structure_id']:r['energy_hartree'] for r in old_records['molecules']}
vectors={}
for name,step in [('best',10),('last',20)]:
    with np.load(ROOT/f'train/{name}.npz',allow_pickle=False) as archive:
        vectors[name]=archive['parameters']
        metadata=json.loads(archive['metadata'].item())
    assert metadata['update_step']==step
initial,predict,backward,mask=make_mace_predictor(0,config,return_mask=True)
ids=['GMTKN55-v2/ALKBDE10/cao','GMTKN55-v2/HAL59/BrBr_OPH3',
     'GMTKN55-v2/HEAVYSB11/ge2h6','GMTKN55-v2/RG18/kr2','GMTKN55-v2/ADIM6/AM2']
result=dict(status='running',passed=False,source_sha256=source,
    parent_source_sha256='37f917b3d144c8545f0c024910e0f5fa47ae3c753cf0d0ddbae4b85b8bb88ca6',
    native_sha256=file_hash(register_integrals()._name),device=jax.devices('gpu')[0].device_kind,
    best_parameter_sha256=hashlib.sha256(vectors['best'].tobytes()).hexdigest(),
    latest_parameter_sha256=hashlib.sha256(vectors['last'].tobytes()).hexdigest(),checks=[])
atomic_json(ROOT/'cao-restart-verification.json',result)
for identifier in ids:
    row=next(r for r in data['structures'] if r['structure_id']==identifier)
    with jax.default_device(jax.devices('cpu')[0]):
        layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
            basis_family=data['basis_family'],core_primitives=data['core_primitives'])
        layout=bind_pool_candidate(layout,candidate)
    free,_=prepare_fixed_pool(row,layout,cache_location(ROOT/'cache',row),storage='device')
    graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=config['elements'],cutoff=config['cutoff'])
    for name in (['last','best'] if identifier==ids[0] else ['best']):
        tick=time.perf_counter();vector=jnp.asarray(vectors[name]);c=predict(vector,graph)
        energy,g,info=free.evaluate_device(c)
        entry=dict(structure_id=identifier,checkpoint=name,energy_hartree=energy,
            stable=info['stable'],converged=info['converged'],info=info)
        if name=='best':
            entry.update(reference_energy_hartree=reference[identifier],
                replay_passed=bool(np.isclose(energy,reference[identifier],atol=1e-9,rtol=0)))
            assert entry['replay_passed'],entry
        if identifier==ids[0] and name=='last':
            checks=[]
            for seed in range(3):
                direction=np.random.default_rng(seed).normal(size=c.shape)
                direction/=np.linalg.norm(direction);ad=float(jnp.sum(g*direction))
                for step in [1e-4,3e-5]:
                    plus=free.evaluate_device(c+step*direction)[0]
                    minus=free.evaluate_device(c-step*direction)[0]
                    fd=(plus-minus)/(2*step)
                    checks.append(dict(seed=seed,step=step,ad=ad,fd=fd,error=abs(ad-fd),
                        passed=bool(np.isclose(ad,fd,atol=2e-6,rtol=2e-5))))
            entry['coefficient_checks']=checks
            assert all(x['passed'] for x in checks),checks
            grad=np.asarray(backward(vector,graph,g))
            direction=np.random.default_rng(0).normal(size=vector.shape)*mask
            direction/=np.linalg.norm(direction);ad=float(grad@direction);step=1e-4
            plus=free.evaluate_device(predict(vector+step*direction,graph))[0]
            minus=free.evaluate_device(predict(vector-step*direction,graph))[0];fd=(plus-minus)/(2*step)
            entry['mace_parameter_check']=dict(step=step,ad=ad,fd=fd,error=abs(ad-fd),
                passed=bool(np.isfinite(grad).all() and np.isclose(ad,fd,atol=2e-6,rtol=2e-5)))
            assert entry['mace_parameter_check']['passed'],entry['mace_parameter_check']
        entry['seconds']=time.perf_counter()-tick
        result['checks'].append(entry);atomic_json(ROOT/'cao-restart-verification.json',result)
        print(identifier,name,energy,entry.get('replay_passed'),entry.get('mace_parameter_check'),flush=True)
result.update(status='passed',passed=True);atomic_json(ROOT/'cao-restart-verification.json',result)
