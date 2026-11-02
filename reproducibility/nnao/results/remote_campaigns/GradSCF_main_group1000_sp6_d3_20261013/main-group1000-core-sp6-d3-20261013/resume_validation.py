"""Resume family gates with immutable caches and audited checkpoint migration."""
import hashlib
import json
from pathlib import Path
import sys

import jax
import numpy as np
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys, core_reference_identity
from gradscf.model.nnao.runtime import repository_root, scientific_source_hash, atomic_json
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
from gradscf.model.nnao.workflows.pool_calibration import load_references

ROOT=Path('artifacts/main-group1000-core-sp6-d3-20261013')
family=sys.argv[1]
manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
for name,key in [('references.json','references_sha256'),('pilot.json','pilot_sha256')]:
    assert hashlib.sha256((ROOT/'inputs'/name).read_bytes()).hexdigest()==manifest[key]
data=dict(load_references(ROOT/'inputs/pilot.json'),basis_family=family)
assert {s for r in data['structures'] for s in r['symbols']}==set(manifest['elements'])
assert jax.default_backend()=='gpu',jax.devices()
source=scientific_source_hash(repository_root())
out=ROOT/'validation-ge2h6-fixed'/family
if (out/'passed.json').exists():
    record=json.loads((out/'passed.json').read_text())
    assert record['passed'] and record['scientific_source_sha256']==source
    print('ALREADY_PASSED',json.dumps(record),flush=True)
    sys.exit(0)
channels={}
with jax.default_device(jax.devices('cpu')[0]):
    layouts=[prepare_direct_basis(list(zip(r['symbols'],r['coords_angstrom'])),
        basis_family=family,core_primitives=data['core_primitives']) for r in data['structures']]
    keys=pool_channel_keys(layouts)
    for layout in layouts:
        for atom,l,role,slot,a in zip(layout.shell_atoms,layout.topology.angular_momenta,
                                    layout.roles,layout.slots,layout.parameters.exponents):
            if slot>=0:
                key=(layout.symbols[atom],l,role)
                channels[key]=dict(element=key[0],l=l,role=role,
                    reference_exponents_bohr_minus2=np.asarray(a).tolist(),
                    candidate_exponents_bohr_minus2=np.asarray(a).tolist())
candidate=dict(parameters=dict(keys=keys,values=np.zeros((len(keys),2)).tolist()),
    channels=[channels[k] for k in keys],core_reference=core_reference_identity(layouts[0]))
parent=ROOT/'validation-kr2-fixed'/family
target=out/'training/last.npz'
migrations={}
for name in ('last.npz','best.npz'):
    path=parent/'training'/name
    destination=out/'training'/name
    if destination.exists() or not path.exists():
        continue
    with np.load(path,allow_pickle=False) as saved:
        arrays={k:np.array(saved[k],copy=True) for k in saved.files if k!='metadata'}
        metadata=json.loads(str(saved['metadata'].item()))
    previous=metadata['identity']['scientific_source_sha256']
    expected='c221d9c0de4e944ce4a5972e8fc91c5bcacba4426d9fd11c143cf707a5e52f69'
    assert previous==expected,previous
    metadata['identity']['scientific_source_sha256']=source
    metadata['source_migration']=dict(parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parent_source=previous,optimizer_reset=False,accepted_epoch=metadata['epoch'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream,**arrays,metadata=np.asarray(json.dumps(metadata,sort_keys=True,allow_nan=False)))
    temporary.replace(destination)
    with np.load(destination,allow_pickle=False) as restored:
        for key,array in arrays.items():
            assert restored[key].dtype==array.dtype and restored[key].shape==array.shape
            assert restored[key].tobytes()==array.tobytes()
    # This checkpoint is not validated until run_fixed_pool replays every saved
    # energy/gradient and passes its finite-difference gate under the new source.
    migrations[name]=metadata['source_migration']
if migrations:
    atomic_json(out/'checkpoint-migration.json',migrations)
result=run_fixed_pool(data,candidate,parent/'cache',out/'training',epochs=1,
                      storage='mmap',resume=target.exists())
assert result['status']=='complete' and result['accepted_epoch']==1
record=dict(family=family,passed=True,reference_count=len(data['structures']),
    element_count=len(manifest['elements']),backend=jax.default_backend(),
    resumed_checkpoint=(out/'checkpoint-migration.json').exists(),
    scientific_source_sha256=source)
atomic_json(out/'passed.json',record)
print(json.dumps(record),flush=True)
