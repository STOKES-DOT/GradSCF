"""Prepared action probes at the measured Davidson widths (6 and 12).

Both variants bind the same saved numerical reference. Independent integral
and grid rebuilding is preparation, not end-to-end chemistry. Ten warm
samples per width are synchronized; there are no diagnostic callbacks.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from worker import Recorder, ready, reference

jax.config.update('jax_enable_x64', True)
cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2])
out.mkdir(parents=True,exist_ok=True);log=Recorder(cfg,out)
mf,ref=reference(cfg,log)
saved=np.load(cfg['shared_states'])
ref=replace(ref,**{key:jnp.asarray(saved[key]) for key in
    ('mo_coeff','mo_occ','mo_energy','rdm1','fock_matrix') if key in saved})
from gradscf.tddft.response import build_restricted_tda_operator
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
vind,diagonal,gaps=log.timed('operator_preparation',lambda:build_restricted_tda_operator(
    ref,SemilocalResponseFunctional(cfg['xc']),
    response_kernel_options=dict(two_electron_mode='df',max_memory=4096,direct_workers=cfg['threads'])))
log.add('memory_plan',None,plan=dict(vind.memory_plan._asdict()))
for batch in (6,12):
    x=jnp.asarray(np.random.default_rng(7).normal(size=(batch,gaps.size)))
    start=time.perf_counter();lowered=jax.jit(vind).lower(x)
    log.add('action_lowering',time.perf_counter()-start,batch_size=batch)
    start=time.perf_counter();compiled=lowered.compile();memory=compiled.memory_analysis()
    log.add('action_compile',time.perf_counter()-start,batch_size=batch,
        compiler_memory={k:getattr(memory,k,None) for k in ('argument_size_in_bytes','temp_size_in_bytes','output_size_in_bytes')})
    for repeat in range(11):
        result=log.timed('action_first' if repeat==0 else 'action_warm',lambda:compiled(x),
            repeat,batch_size=batch,notes='fixed saved reference; preparation and compilation excluded')
    np.savez(out/f'action-batch{batch}.npz',x=np.asarray(x),action=np.asarray(result))
(out/'clean_phase_done.json').write_text(json.dumps(dict(ended_monotonic=time.monotonic()))+'\n')
