"""Check the exact failed As2Me4 pool gradient on the larger GPU first."""
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys
from gradscf.model.nnao.training.pool import PoolRHF
from gradscf.model.nnao.runtime import atomic_json,scientific_source_hash,repository_root

root=Path(__file__).resolve().parent
expected=json.loads((root/'source.json').read_text())['source_sha256']
assert scientific_source_hash(repository_root())==expected
data=json.loads((root/'inputs/pilot.json').read_text())
cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
assert jax.config.jax_enable_x64
with jax.default_device(cpu):
    layouts=[prepare_direct_basis(list(zip(r['symbols'],r['coords_angstrom'])),
        basis_family=data['basis_family'],core_primitives=data['core_primitives']) for r in data['structures']]
    keys=pool_channel_keys(layouts)
    zero=jnp.zeros((len(keys),2),dtype=jnp.float64)
row=next(r for r in data['structures'] if r['structure_id']=='GMTKN55-v2/HEAVYSB11/as2me4')
before=gpu.memory_stats();tick=time.perf_counter()
model=PoolRHF(row,keys,basis_family=data['basis_family'],core_primitives=data['core_primitives'],scf_device=gpu)
energy,gradient,info=model.evaluate(zero)
assert np.isfinite(gradient).all()
result=dict(passed=True,structure_id=row['structure_id'],source_sha256=expected,device=gpu.device_kind,
    energy_hartree=energy,gradient_norm=float(np.linalg.norm(gradient)),info=info,
    memory_before=before,memory_after=gpu.memory_stats(),elapsed_seconds=time.perf_counter()-tick)
atomic_json(root/'as2me4-80gb-probe.json',result)
print(json.dumps(result,indent=2),flush=True)
