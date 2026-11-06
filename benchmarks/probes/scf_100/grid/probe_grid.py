"""Actual 96-atom grid: compile, materialization, cache read and XC diagnostics."""
import argparse,json,os,time,statistics,hashlib,resource
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args()
assert os.getenv('SLURM_JOB_ID')
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:64])
os.environ.update(JAX_PLATFORMS='cpu' if a.device=='cpu' else 'cuda,cpu',JAX_ENABLE_X64='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_ENABLE_COMPILATION_CACHE='false')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
a.output.mkdir(parents=True,exist_ok=False)
import numpy as np
import jax
import jax.numpy as jnp
from gradscf import gto
from gradscf.integrals.basis import basis_from_molecule_spec
from gradscf.integrals.grids import build_molecular_grid_from_spec
from gradscf.integrals.grids.blocks import AOGrid,_materialize_scf_grid
from gradscf.scf.rks import RKSConfig,_restricted_xc_energy_and_matrix
root=Path('/home/user_yuanjiao/private/homefile/gradscf-scf-100-20261105')
config=root/'baseline/reports/scf_100/20261105-40a569bf/plan-v2/configs/beta-carotene-def2-svp-rks-pbe-direct-gradscf.json'
cfg=json.loads(config.read_text());cpu=jax.devices('cpu')[0];target=jax.devices(a.device)[0]
def save(name,value): (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
rows=[]
def emit(row):
    rows.append(row);save('measurements.json',rows);print(json.dumps(row),flush=True)
def sync(x):return jax.block_until_ready(x)
def benchmark(name,fn,args,repeat=3):
    sync(args);tick=time.perf_counter();lowered=fn.lower(*args);lower=time.perf_counter()-tick
    tick=time.perf_counter();ex=lowered.compile();compile_s=time.perf_counter()-tick
    text=ex.as_text();(a.output/(name+'.hlo.txt')).write_text(text)
    m=ex.memory_analysis();samples=[]
    for _ in range(repeat+1):
        tick=time.perf_counter();value=sync(ex(*args));samples.append(time.perf_counter()-tick)
    emit(dict(stage=name,lower_seconds=lower,compile_seconds=compile_s,first_seconds=samples[0],warm_seconds=samples[1:],warm_median=statistics.median(samples[1:]),
        ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        xla_memory={k:getattr(m,k,None) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')},
        gpu_allocator=target.memory_stats() if a.device=='gpu' else None))
    return value
with jax.default_device(cpu):
    tick=time.perf_counter();mol=gto.M(atom=cfg['atom'],basis=cfg['basis'],unit=cfg['unit'],cart=True)
    spec=mol.to_spec();basis=basis_from_molecule_spec(spec,basis=cfg['basis'],max_l=3)
    coords,weights=build_molecular_grid_from_spec(spec,level=3)
    raw=sync(AOGrid.from_basis(basis,coords,weights,block_size=4096,adaptive=True))
    emit(dict(stage='basis_grid_prepare',seconds=time.perf_counter()-tick,nao=raw.nao,ngrid=raw.ngrids))
    assert raw.nao==880 and raw.ngrids==1113968
with jax.default_device(target):
    raw=sync(jax.device_put(raw,target))
    save('manifest.json',dict(job=os.environ['SLURM_JOB_ID'],device=str(target),device_kind=target.device_kind,jax=jax.__version__,affinity=sorted(os.sched_getaffinity(0)),dtype='float64',
        ao_source=str(Path(__import__('gradscf.integrals.grids.ao',fromlist=['x']).__file__)),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
    # One compiled dynamic-index kernel at three actual grid positions.
    fn=jax.jit(lambda r,i:r.block(i,4096,deriv=1)[1])
    for start in (0,(raw.ngrids//8192)*4096,raw.ngrids-4096):
        benchmark('block_'+str(start),fn,(raw,jnp.asarray(start)),repeat=3)
    if a.device=='cpu':
        material=jax.jit(lambda r:_materialize_scf_grid(r,block_size=4096,deriv=1))
        cache=benchmark('materialize',material,(raw,))
        indices=np.unique(np.linspace(0,raw.ngrids-1,257,dtype=int))
        np.savez(a.output/'ao_samples.npz',indices=indices,values=np.asarray(cache.values[indices]))
    else:
        cache=raw
    with np.load(root/'stage-cpu-gpu-20261106/diagnostic-gpu/shared_state.npz') as f:dm=jax.device_put(f['density'],target)
    def xc(d,r):return _restricted_xc_energy_and_matrix(density=d,ao=jnp.zeros((0,raw.nao)),ao_deriv1=jnp.zeros((4,0,raw.nao)),
        weights=r.weights,cfg=RKSConfig(xc_spec='pbe'),xc_kind='GGA',grid_ao=r)
    energy,vmat=benchmark('whole_grid_xc',jax.jit(xc),(dm,cache))
    np.savez(a.output/'xc.npz',energy=np.asarray(energy),vxc=np.asarray(vmat))
    save('complete.json',dict(status='ok',scope='shared-input AO/XC kernels, not a complete SCF',energy=float(energy)))
