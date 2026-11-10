"""Fixed-basis direct J/K before/after guard; no SCF or AO ERI construction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'density', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('At least3 warm repetitions are required')
    if (not os.getenv('SLURM_JOB_ID') or not 1 <= args.workers <= 64
            or len(os.sched_getaffinity(0)) < args.workers):
        parser.error('Use allocated CPUs within the authorized64 logical-CPU limit')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.workers])
    os.environ.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    import jax
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    from gradscf import integrals
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    from gradscf.integrals._native import register_integrals
    args.output.mkdir(parents=True, exist_ok=False)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cfg=json.loads(args.config.read_text())
    start=time.perf_counter()
    top,p=integrals.prepare_basis(cfg['atom'],cfg['basis'],unit=cfg['unit'],
        cart=cfg['cart'],charge=cfg['charge'],spin=cfg['spin'])
    provider=NativeDirectBasis(integrals.make_plan(top),p,private_output_bytes=1024**3)
    with np.load(args.density,allow_pickle=False) as state:
        density=jnp.asarray(state['density'])
    if density.shape != (top.nao,top.nao) or density.dtype != jnp.float64:
        raise ValueError('Expected complete float64 spatial density')
    jax.block_until_ready((density,provider))
    prepared=time.perf_counter()-start
    library=Path(register_integrals()._name).resolve()
    runtime=dict(job=os.environ['SLURM_JOB_ID'],host=platform.node(),python=sys.executable,
        module=gradscf.__file__,jax=jax.__version__,dtype=str(density.dtype),nao=top.nao,
        config_sha256=digest(args.config),density_sha256=digest(args.density),script_sha256=digest(__file__),
        native=str(library),native_sha256=digest(library),affinity=sorted(os.sched_getaffinity(0)),
        workers=args.workers,warm_repetitions=args.repeats,private_output_bytes=1024**3,cutoff=1e-12,omega=0.,preparation_seconds=prepared,
        scope='Saved GradSCF density, fixed basis, selected J and K; CPU integral/contraction kernel only',
        forward_gate='same-output atol1e-10 rtol1e-12; warm median regression target <=10%, investigate larger drift')
    (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    rows=[]
    for screened in (False,True):
        name='density_screened' if screened else 'basis_screened'
        def action(d):
            return provider.get_jk(d,workers=args.workers,screening_threshold=1e-12,
                screen_density=density if screened else None)
        start=time.perf_counter();lowered=jax.jit(action).lower(density);lower=time.perf_counter()-start
        start=time.perf_counter();compiled=lowered.compile();compile=time.perf_counter()-start
        start=time.perf_counter();value=jax.block_until_ready(compiled(density));first=time.perf_counter()-start
        host=[np.asarray(v) for v in value]
        assert all(np.isfinite(v).all() for v in host)
        difference=None
        if args.reference:
            old=json.loads((args.reference/'runtime.json').read_text())
            for key in ('config_sha256','density_sha256','nao','workers','private_output_bytes','cutoff','omega','dtype'):
                assert old[key]==runtime[key],key
            with np.load(args.reference/(name+'.npz')) as previous:
                for key,current in zip(('j','k'),host):
                    np.testing.assert_allclose(current,previous[key],atol=1e-10,rtol=1e-12)
                difference=max(float(np.max(abs(host[i]-previous[k]))) for i,k in enumerate(('j','k')))
        np.savez(args.output/(name+'.npz'),j=host[0],k=host[1])
        samples=[]
        for _ in range(args.repeats):
            start=time.perf_counter();jax.block_until_ready(compiled(density));samples.append(time.perf_counter()-start)
        memory=compiled.memory_analysis()
        row=dict(variant=name,lower_seconds=lower,compile_seconds=compile,
            first_compiled_seconds=first,warm_seconds=samples,warm_median_seconds=statistics.median(samples),
            maxabs_reference_error=difference,ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            xla_temp_bytes=None if memory is None else memory.temp_size_in_bytes)
        rows.append(row);print(json.dumps(row),flush=True)
        (args.output/'summary.json').write_text(json.dumps(dict(complete=len(rows)==2,records=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
