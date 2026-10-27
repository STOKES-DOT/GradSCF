"""Isolated packed scheduling / memory probes, no validation oracle in peak."""
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from worker import Recorder,ready

jax.config.update('jax_enable_x64',True)
cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
log=Recorder(cfg,out);n=cfg['nao'];batch=cfg.get('batch',5)
log.context.update(nao=n,batch_size=batch,nroots=None)
rng=np.random.default_rng(7);pairs=n*(n+1)//2
raw=rng.normal(size=(pairs,12))*.01
device=jax.devices(cfg['device'])[0]
started=time.perf_counter()
host_inputs=(raw@raw.T,rng.normal(size=(batch,n,n)))
log.add('synthetic_input_generation',time.perf_counter()-started,notes='not Gaussian integrals')
started=time.perf_counter()
eri,density=ready(jax.device_put(host_inputs,device))
log.add('input_transfer',time.perf_counter()-started,input_bytes=eri.nbytes+density.nbytes)
if cfg.get('implementation')=='td-private':
    from gradscf.tddft.response import _jk_from_eri_pair_matrix as action
else:
    from gradscf.integrals.molecular.jk import build_jk_from_packed
    opts={k:cfg[k] for k in ('max_memory','block_size') if k in cfg}
    action=lambda e,d:build_jk_from_packed(e,d,**opts)
started=time.perf_counter();lowered=jax.jit(action).lower(eri,density)
log.add('lower',time.perf_counter()-started)
started=time.perf_counter();compiled=lowered.compile()
mem=compiled.memory_analysis()
log.add('compile',time.perf_counter()-started,compiler_memory={k:getattr(mem,k,None) for k in
    ('argument_size_in_bytes','temp_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','host_temp_size_in_bytes')})
(out/'optimized.hlo.txt').write_text(compiled.as_text())
for repeat in range(11):
    value=log.timed('first_action' if repeat==0 else 'warm_action',lambda:compiled(eri,density),repeat)
if cfg['device']=='gpu':
    log.add('allocator',None,allocator_stats=device.memory_stats(),notes='in-use and pool peaks are distinct; no NVML sampling')
(out/'clean_phase_done.json').write_text(json.dumps(dict(ended_monotonic=time.monotonic()))+'\n')
np.savez(out/'values.npz',j=np.asarray(value[0]),k=np.asarray(value[1]))
