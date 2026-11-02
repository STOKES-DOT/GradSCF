"""Validate explicit input identity and one real GPU update for each family."""
import hashlib
import json
from pathlib import Path
import sys

import jax
import numpy as np

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys
from gradscf.model.nnao.runtime import repository_root, scientific_source_hash
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
from gradscf.model.nnao.workflows.pool_calibration import load_references

ROOT = Path('artifacts/main-group1000-restart-20261013')
family = sys.argv[1]
manifest = json.loads((ROOT/'inputs/manifest.json').read_text())
for name, key in [('references.json','references_sha256'),('pilot.json','pilot_sha256')]:
    assert hashlib.sha256((ROOT/'inputs'/name).read_bytes()).hexdigest() == manifest[key]
data = dict(load_references(ROOT/'inputs/pilot.json'), basis_family=family)
assert {s for row in data['structures'] for s in row['symbols']} == set(manifest['elements'])
assert jax.default_backend() == 'gpu', jax.devices()
channels = {}
with jax.default_device(jax.devices('cpu')[0]):
    layouts = [prepare_direct_basis(list(zip(r['symbols'],r['coords_angstrom'])),
        basis_family=family, core_primitives='reference') for r in data['structures']]
    keys = pool_channel_keys(layouts)
    for layout in layouts:
        for atom,l,role,slot,a in zip(layout.shell_atoms,layout.topology.angular_momenta,
                                     layout.roles,layout.slots,layout.parameters.exponents):
            if slot >= 0:
                key=(layout.symbols[atom],l,role)
                channels[key]=dict(element=key[0],l=l,role=role,
                    reference_exponents_bohr_minus2=np.asarray(a).tolist(),
                    candidate_exponents_bohr_minus2=np.asarray(a).tolist())
candidate = dict(parameters=dict(keys=keys,values=np.zeros((len(keys),2)).tolist()),
                 channels=[channels[k] for k in keys])
out = ROOT/'validation'/family
result = run_fixed_pool(data,candidate,out/'cache',out/'training',epochs=1,storage='mmap')
assert result['status']=='complete' and result['accepted_epoch']==1
record=dict(family=family,passed=True,reference_count=len(data['structures']),
    element_count=len(manifest['elements']),backend=jax.default_backend(),
    scientific_source_sha256=scientific_source_hash(repository_root()))
(out/'passed.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
