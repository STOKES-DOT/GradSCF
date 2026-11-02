"""Numerical gate for the fixed-pool atomic-population fallback and its implicit AD."""
from pathlib import Path
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
from gradscf.model.nnao.runtime import atomic_json,cache_location,scientific_source_hash,repository_root

ROOT=Path(__file__).resolve().parent
source=json.loads((ROOT/'source.json').read_text())['source_sha256']
assert scientific_source_hash(repository_root())==source
data=json.loads((ROOT/'inputs/references.json').read_text())
config=json.loads((ROOT/'inputs/model-config.json').read_text())
candidate=json.loads((ROOT/'selection/candidate.json').read_text())
ids=['GMTKN55-v2/HAL59/BrBr_OPH3','GMTKN55-v2/ALKBDE10/cao',
     'GMTKN55-v2/HEAVYSB11/ge2h6','GMTKN55-v2/RG18/kr2']
rows=[next(r for r in data['structures'] if r['structure_id']==name) for name in ids]
rows.append(next(r for r in data['structures'] if r['natoms']<=10 and set(r['symbols'])<=set('HCNOF')))
initial,predict,backward,mask=make_mace_predictor(0,config,return_mask=True)
cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
result=dict(status='running',source_sha256=source,device=gpu.device_kind,results=[])
for row in rows:
    tick=time.perf_counter()
    with jax.default_device(cpu):
        layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
            basis_family=data['basis_family'],core_primitives=data['core_primitives'])
        layout=bind_pool_candidate(layout,candidate)
    free,_=prepare_fixed_pool(row,layout,cache_location(ROOT/'cache',row),storage='device')
    graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=config['elements'])
    c=predict(initial,graph)
    energy,gradient,info=free.evaluate_device(c)
    direction=np.random.default_rng(0).normal(size=c.shape);direction/=np.linalg.norm(direction)
    ad=float(jnp.sum(gradient*direction));checks=[]
    for h in [1e-4,3e-5]:
        plus=free.evaluate_device(c+h*direction,gradient=False)[0]
        minus=free.evaluate_device(c-h*direction,gradient=False)[0]
        fd=(plus-minus)/(2*h)
        checks.append(dict(step=h,ad=ad,fd=fd,error=abs(ad-fd),passed=bool(np.isclose(ad,fd,atol=2e-6,rtol=2e-5))))
    assert all(v['passed'] for v in checks),checks
    entry=dict(structure_id=row['structure_id'],energy_hartree=energy,info=info,
               coefficient_checks=checks,seconds=time.perf_counter()-tick)
    if row['structure_id']==ids[0]:
        np.testing.assert_allclose(energy,-5545.141625904812,atol=1e-7,rtol=0)
        g=np.asarray(backward(initial,graph,gradient))
        d=np.random.default_rng(0).normal(size=initial.shape)*mask;d/=np.linalg.norm(d)
        ad=float(g@d);h=1e-4
        plus=free.evaluate_device(predict(initial+h*d,graph),gradient=False)[0]
        minus=free.evaluate_device(predict(initial-h*d,graph),gradient=False)[0]
        fd=(plus-minus)/(2*h)
        entry['mace_parameter_check']=dict(ad=ad,fd=fd,error=abs(ad-fd),passed=bool(np.isclose(ad,fd,atol=2e-6,rtol=2e-5)))
        assert entry['mace_parameter_check']['passed'],entry['mace_parameter_check']
    result['results'].append(entry);atomic_json(ROOT/'atomic-rescue-verification.json',result)
    print(row['structure_id'],energy,checks,flush=True)
result.update(status='passed',passed=True)
atomic_json(ROOT/'atomic-rescue-verification.json',result)
