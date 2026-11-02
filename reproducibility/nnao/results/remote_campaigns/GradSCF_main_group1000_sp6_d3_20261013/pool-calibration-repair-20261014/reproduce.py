"""Observe the original calibration ensemble without changing its numerics."""
import json
import os
from pathlib import Path
import time
import sys
import jax
import jax.numpy as jnp
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys
from gradscf.model.nnao.training.pool import PoolRHF
from gradscf.model.nnao.training.pool import overlap_diagnostics
from gradscf.model.nnao.training.objective import BasisRHF

if '--gpu' in sys.argv:
    cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
    def hybrid_evaluate(self,parameters,*,gradient=True):
        parameters=jax.device_put(parameters,cpu)
        physical=jax.block_until_ready(self.physical(parameters))
        condition=overlap_diagnostics(physical[0],max_condition=self.max_condition)
        physical=jax.device_put(physical,gpu);enuc=jax.device_put(self.enuc,gpu)
        if gradient:
            (energy,info),cotangent=self.gradient(physical,enuc)
        else:energy,info=self.value(physical,enuc)
        details=BasisRHF._checked_info(energy,info)
        derivative=np.asarray(self.pullback(parameters,jax.device_put(cotangent,cpu))) if gradient else None
        if derivative is not None and not np.isfinite(derivative).all():raise RuntimeError('Nonfinite exponent gradient')
        return float(energy),derivative,dict(details,**condition)
    import numpy as np
    PoolRHF.evaluate=hybrid_evaluate

root=Path('artifacts/pool-calibration-repair-20261014')
data=json.loads(Path('artifacts/main-group1000-core-sp6-d3-20261013/inputs/pilot.json').read_text())
family='nnao_dzp_s'
def report(stage,**extra):
    status=Path('/proc/self/status').read_text().splitlines()
    print(json.dumps(dict(stage=stage,**extra,
        maps=len(Path('/proc/self/maps').read_text().splitlines()),
        memory={s.split(':')[0]:s.split(':',1)[1].strip() for s in status if s.startswith(('VmRSS:','VmHWM:','RssAnon:','Threads:'))})),flush=True)
report('start',pid=os.getpid(),affinity=len(os.sched_getaffinity(0)),
       map_limit=Path('/proc/sys/vm/max_map_count').read_text().strip())
rows=data['structures']
with jax.default_device(jax.devices('cpu')[0]):
    layouts=[prepare_direct_basis(list(zip(r['symbols'],r['coords_angstrom'])),
        basis_family=family,core_primitives=data['core_primitives']) for r in rows]
    keys=pool_channel_keys(layouts)
    models=[PoolRHF(r,keys,basis_family=family,core_primitives=data['core_primitives']) for r in rows]
report('plans_ready')
parameters=jax.device_put(jnp.zeros((len(keys),2),dtype=jnp.float64),jax.devices('cpu')[0])
for index,model in enumerate(models):
    report('evaluate_begin',index=index,structure=model.row['structure_id'],pool_nao=model.pool_nao)
    start=time.perf_counter()
    energy,gradient,info=model.evaluate(parameters)
    report('evaluate_end',index=index,structure=model.row['structure_id'],
        energy=energy,gradient_norm=float(jnp.linalg.norm(gradient)),seconds=time.perf_counter()-start)
