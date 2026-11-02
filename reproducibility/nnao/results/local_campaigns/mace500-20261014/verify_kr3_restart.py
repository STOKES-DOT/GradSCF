"""Forward, analytic backward and checkpoint replay only; no finite differences."""
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
model=json.loads((ROOT/'inputs/model-config.json').read_text())
pool=json.loads((ROOT/'selection/candidate.json').read_text())
records=json.loads((ROOT/'train/epoch-molecules.json').read_text())
assert records['epoch']==2
reference={x['structure_id']:x['energy_hartree'] for x in records['molecules']}
vectors={}
for name,file,step in [('last','last.npz',29),('best','best.npz',20),('failed','failed-update-000030.npz',29)]:
    with np.load(ROOT/'train'/file,allow_pickle=False) as a:
        vectors[name]=a['parameters'];metadata=json.loads(a['metadata'].item())
    assert metadata.get('update_step',metadata.get('accepted_updates'))==step
_,predict,backward,_=make_mace_predictor(0,model,return_mask=True)
ids=['GMTKN55-v2/RG18/kr3','GMTKN55-v2/ALKBDE10/cao','GMTKN55-v2/HAL59/BrBr_OPH3']
report=dict(status='running',passed=False,finite_differences=False,source_sha256=source,
    parent_source_sha256='0a5510b43d00e003eb6d0f16258d48d14a5553d05bfda5b960c6467da85fefc5',
    native_sha256=file_hash(register_integrals()._name),device=jax.devices('gpu')[0].device_kind,
    parameter_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in vectors.items()},checks=[])
atomic_json(ROOT/'kr3-restart-verification.json',report)
for identifier in ids:
    row=next(r for r in data['structures'] if r['structure_id']==identifier)
    with jax.default_device(jax.devices('cpu')[0]):
        layout=bind_pool_candidate(prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
            basis_family=data['basis_family'],core_primitives=data['core_primitives']),pool)
    free,_=prepare_fixed_pool(row,layout,cache_location(ROOT/'cache',row),storage='device')
    graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=model['elements'],cutoff=model['cutoff'])
    for name in (['failed','last','best'] if identifier==ids[0] else ['last','best']):
        tick=time.perf_counter();vector=jnp.asarray(vectors[name]);c=predict(vector,graph)
        energy,g,info=free.evaluate_device(c)
        network_gradient=backward(vector,graph,g)
        entry=dict(structure_id=identifier,checkpoint=name,energy_hartree=energy,converged=info['converged'],
            stable=info['stable'],finite_gradient=bool(jnp.all(jnp.isfinite(g))),
            finite_network_gradient=bool(jnp.all(jnp.isfinite(network_gradient))),info=info)
        assert entry['finite_gradient'] and entry['finite_network_gradient']
        if name=='best':
            entry.update(reference_energy_hartree=reference[identifier],
                replay_passed=bool(np.isclose(energy,reference[identifier],atol=1e-9,rtol=0)))
            assert entry['replay_passed'],entry
        entry['seconds']=time.perf_counter()-tick
        report['checks'].append(entry);atomic_json(ROOT/'kr3-restart-verification.json',report)
        print(identifier,name,energy,info['stability_eigen_residual'],info.get('stability_dense_fallback'),flush=True)
report.update(status='passed',passed=True);atomic_json(ROOT/'kr3-restart-verification.json',report)
