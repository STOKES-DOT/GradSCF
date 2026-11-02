"""Real GPU MACE -> fixed-pool contraction -> implicit SCF for one family."""
import json
from pathlib import Path
import sys
import jax
import numpy as np

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys
from gradscf.model.nnao.workflows.pool_calibration import load_references
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool

family = sys.argv[1]
assert jax.default_backend() == 'gpu', jax.devices()
data = dict(load_references(ids=['CH4', 'NH3', 'H2O', 'HF', 'CO']), basis_family=family)
with jax.default_device(jax.devices('cpu')[0]):
    layouts = [prepare_direct_basis(list(zip(r['symbols'], r['coords_angstrom'])),
                                   basis_family=family) for r in data['structures']]
    keys = pool_channel_keys(layouts)
    channels = {}
    for layout in layouts:
        for atom, l, role, slot, a in zip(layout.shell_atoms, layout.topology.angular_momenta,
                                         layout.roles, layout.slots, layout.parameters.exponents):
            if slot >= 0:
                key = (layout.symbols[atom], l, role)
                channels[key] = dict(element=key[0], l=l, role=role,
                    reference_exponents_bohr_minus2=np.asarray(a).tolist(),
                    candidate_exponents_bohr_minus2=np.asarray(a).tolist())
candidate = dict(parameters=dict(keys=keys, values=np.zeros((len(keys), 2)).tolist()),
                 channels=[channels[k] for k in keys])
root = Path('artifacts/basis-grid-20261013/smoke')/family
result = run_fixed_pool(data, candidate, root/'cache', root/'training', epochs=1)
assert result['status'] == 'complete'
if family == 'nnao_tzp_l':
    result = run_fixed_pool(data, candidate, root/'cache', root/'training', epochs=2, resume=True)
    assert result['accepted_epoch'] == 2
print(json.dumps(dict(family=family, status=result['status'], epoch=result['accepted_epoch'])), flush=True)
