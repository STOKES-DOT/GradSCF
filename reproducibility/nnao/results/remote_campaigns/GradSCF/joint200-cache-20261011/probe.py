"""Measure compiler mappings in the failed 200-structure engine, without updates."""
import gc,importlib.util,json,os,time,resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax

root=Path.cwd();task=Path(os.environ['CACHE_TASK']);out=task/f'probe-{os.environ["SLURM_JOB_ID"]}'
out.mkdir(parents=True,exist_ok=True)
parent=Path('/home/user_yuanjiao/private/homefile/GradSCF/artifacts/nnao-joint200-20261011')
spec=importlib.util.spec_from_file_location('cache_probe_driver',root/'tools/train_nnao_joint_ad.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
limit=int(Path('/proc/sys/vm/max_map_count').read_text());events=[]

def record(phase,**extra):
    status=dict(x.split(':',1) for x in Path('/proc/self/status').read_text().splitlines() if ':' in x)
    item=dict(phase=phase,time=time.time(),maps=len(Path('/proc/self/maps').read_text().splitlines()),
              map_limit=limit,rss_kib=int(status['VmRSS'].split()[0]),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**extra)
    events.append(item)
    with (out/'events.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    print(json.dumps(item),flush=True)
    return item

rows=m.base.pilot.read_structures(parent/'train.jsonl')
theta,beta,metadata=m.read_initial_parameters(parent/'initial.npz',log_bound=.25)
args=SimpleNamespace(platform='cuda',auxbasis='def2-universal-jkfit',implicit_tolerance=1e-8,
                     scf_rescue_level_shift=.2,output_dir=out,log_bound=.25)
record('before-engine')
engine=m.NativeJointEngine(rows,args,metadata['identity'],None)
record('after-engine')
vector=np.concatenate((theta,beta));cpu=[]
for context in engine.contexts.values():
    cpu.extend([context['physical'],context['pullback']])

def clear_cpu():
    for function in cpu:function.clear_cache()
    gc.collect()
    return record('after-selective-cpu-clear')

def measured(key):
    record('before-molecule',structure_id=key)
    tick=time.perf_counter();e,g,info=engine.molecule(vector,key)
    assert np.isfinite(e) and np.isfinite(g).all() and info['stable']
    state=record('after-molecule',structure_id=key,energy=e,seconds=time.perf_counter()-tick,
        gpu_predict_cache=engine.predict._cache_size(),gpu_backward_cache=engine.backward._cache_size())
    return e,g,state

# Stop well before the kernel map-count limit rather than crashing again.
for row in rows[:36]:
    if record('headroom-check')['maps']>int(.72*limit):break
    measured(row['structure_id'])
record('before-selective-cpu-clear')
cleared=clear_cpu()
if cleared['maps']>int(.60*limit):
    jax.clear_caches();gc.collect();record('after-global-clear')
key='S22/19a'
e,g,_=measured(key)
clear_cpu()
e2,g2,_=measured(key)
np.testing.assert_allclose(e,e2,atol=1e-10,rtol=0)
np.testing.assert_allclose(g,g2,atol=2e-9,rtol=2e-7)
(out/'summary.json').write_text(json.dumps(dict(status='complete',target=key,energy=e,
    gradient_repeat_max_error=float(np.max(np.abs(g-g2))),events=events),indent=2))
