"""Shared-input CPU RI-J/whitening diagnostic; never an end-to-end SCF timer."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('density', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'): parser.error('Numerical probes require a Slurm step')
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 64: parser.error('This fixed probe requires 64 allocated logical CPUs')
    os.sched_setaffinity(0, cpus[:64])
    os.environ.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', GRADSCF_NATIVE_THREADS='64')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    from benchmarks.probes.scf_100.worker import validate_config
    validate_config(cfg)
    if cfg['engine'] != 'gradscf' or cfg['representation'] != 'df': parser.error('Use a frozen GradSCF RI configuration')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.scipy.linalg import cho_solve
    import gradscf
    from gradscf.integrals import prepare_basis, make_auxiliary_plan
    from gradscf.integrals._native import register_integrals
    from gradscf.integrals.molecular.density_fitting import unpack_factors
    from gradscf.integrals.molecular.jk import build_j_from_df
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        threadpool_info, threadpool_limits = lambda: None, None
    jax.config.update('jax_enable_x64', True)
    native = Path(register_integrals()._name).resolve()
    package = Path(gradscf.__file__).parent
    metadata = dict(scope='shared-input/kernel; PySCF density generation excluded; no SCF or geometry AD',
        python=sys.executable, module=gradscf.__file__, host=platform.node(), job=os.environ['SLURM_JOB_ID'],
        affinity=sorted(os.sched_getaffinity(0)), jax=jax.__version__, numpy=np.__version__,
        native=str(native), native_sha256=digest(native), probe_sha256=digest(__file__),
        config_sha256=digest(args.config), density_sha256=digest(args.density),
        source_sha256={name:digest(package/name) for name in ('integrals/molecular/density_fitting.py','integrals/molecular/jk.py')},
        initial_threadpools=threadpool_info(), dtype='float64', device='cpu', units='seconds,bytes,Hartree')
    (args.output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    def emit(stage, **fields):
        row = dict(stage=stage, ru_maxrss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                   rss_kind='cumulative whole-process peak, not stage peak', **fields)
        with (args.output/'measurements.jsonl').open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    def timed(function, *values):
        t = time.perf_counter(); result = function(*values); jax.block_until_ready(result)
        return result, time.perf_counter()-t
    def lower_density(d):
        i, j = jnp.tril_indices(d.shape[0])
        return d[i,j]+d[j,i]-jnp.where(i == j, d[i,j], 0.)
    def expand(pair, n):
        row, col = jnp.arange(n)[:,None], jnp.arange(n)[None,:]
        hi, lo = jnp.maximum(row,col), jnp.minimum(row,col)
        return pair[hi*(hi+1)//2+lo]
    def packed_j(l, d):
        return expand(l.T @ (l @ lower_density(d)), d.shape[0])
    def raw_j(c, chol, d):
        return expand(c.T @ cho_solve((chol, True), c @ lower_density(d)), d.shape[0])
    # Small nonsymmetric density and packed-factor derivative oracle: no native calls.
    rng = np.random.default_rng(20261105)
    tiny = jnp.asarray(rng.normal(size=(3,10))*.1); td = jnp.asarray(rng.normal(size=(4,4)))
    dl, dd = jnp.asarray(rng.normal(size=tiny.shape)*.03), jnp.asarray(rng.normal(size=td.shape)*.03)
    def oracle(l, d):
        f = jax.vmap(lambda p: expand(p, d.shape[0]))(l)
        eri = jnp.einsum('Qpq,Qrs->pqrs', f, f)
        return jnp.einsum('pqrs,sr->pq', eri, d)
    value, tangent = jax.jvp(packed_j, (tiny,td), (dl,dd))
    reference, reference_tangent = jax.jvp(oracle, (tiny,td), (dl,dd))
    weight = jnp.asarray(rng.normal(size=(4,4)))
    gradients = jax.grad(lambda l,d: jnp.sum(packed_j(l,d)*weight), argnums=(0,1))(tiny,td)
    expected = jax.grad(lambda l,d: jnp.sum(oracle(l,d)*weight), argnums=(0,1))(tiny,td)
    for actual, target in [(value,reference),(tangent,reference_tangent),*zip(gradients,expected)]:
        np.testing.assert_allclose(actual, target, atol=1e-10, rtol=1e-10)
    emit('small_array_ad', status='ok', jvp_error=float(jnp.max(jnp.abs(tangent-reference_tangent))),
         vjp_error=max(float(jnp.max(jnp.abs(a-b))) for a,b in zip(gradients,expected)),
         coverage='packed factors and nonsymmetric density arrays; not full nuclear/basis/SCF response')
    t = time.perf_counter()
    opts = dict(unit=cfg['unit'], cart=cfg['cart'], charge=cfg['charge'], spin=cfg['spin'])
    top, params = prepare_basis(cfg['atom'], cfg['basis'], **opts)
    aux, ap = prepare_basis(cfg['atom'], cfg['auxbasis'], **opts)
    plan = make_auxiliary_plan(top, aux)
    emit('basis_plans', seconds=time.perf_counter()-t, nao=top.nao, naux=aux.nao)
    density_host = np.load(args.density, allow_pickle=False)
    if density_host.shape != (top.nao,top.nao) or density_host.dtype != np.float64:
        raise ValueError('The saved reference density must be a spatial float64 (nao,nao) array')
    density, transfer_seconds = timed(jnp.asarray, density_host)
    emit('density_to_jax', seconds=transfer_seconds, bytes=int(density_host.nbytes))
    (metric, raw), seconds = timed(plan.evaluate, params, ap)
    emit('native_2c_3c', seconds=seconds, metric_bytes=int(metric.nbytes), packed_c_bytes=int(raw.nbytes))
    t = time.perf_counter(); factor = plan._factor_metric(metric, plan._signature(params,ap), cfg['df_tol'])
    emit('metric_factor', seconds=time.perf_counter()-t, method=factor.method, shape=list(factor.matrix.shape))
    primary = None
    for threads in (1, 8):
        if threads == 8 and not threadpool_limits:
            emit('host_whiten_blas8', status='unsupported', notes='threadpoolctl is not installed'); continue
        samples = []
        context = threadpool_limits(limits=threads, user_api='blas') if threadpool_limits else nullcontext()
        with context:
            for repeat in range(3):
                t = time.perf_counter(); current = factor.whiten(raw)
                seconds = time.perf_counter()-t; samples.append(seconds)
                if primary is None: primary = current
                error, equal = 0., True
                if current is not primary:
                    for first in range(0, primary.shape[0], 32):
                        a, b = primary[first:first+32], current[first:first+32]
                        error = max(error, float(np.max(np.abs(a-b))))
                        equal &= bool(np.allclose(a,b,atol=1e-10,rtol=1e-10))
                    del a, b
                emit('host_whiten_blas'+str(threads), repeat=repeat, seconds=seconds,
                     status='ok' if equal else 'numerical_mismatch', max_abs_difference=error,
                     threadpools=threadpool_info(), shape=list(current.shape), bytes=int(current.nbytes),
                     c_contiguous=bool(current.flags.c_contiguous), f_contiguous=bool(current.flags.f_contiguous),
                     includes='input/output finite checks and host solve; comparison excluded')
                del current; gc.collect()  # Keep only first baseline and at most one current temporary.
        emit('host_whiten_summary', blas_threads=threads, seconds=samples,
             median_seconds=float(np.median(samples)), spread_seconds=float(np.ptp(samples)))
    packed, seconds = timed(jnp.asarray, primary)
    emit('packed_l_to_jax', seconds=seconds, bytes=int(packed.nbytes))
    del primary; gc.collect()
    full, seconds = timed(unpack_factors, packed, top.nao)
    emit('full_l_unpack', seconds=seconds, bytes=int(full.nbytes))
    variants = [('full_l',build_j_from_df,(full,density)), ('packed_l',packed_j,(packed,density))]
    if factor.method == 'cholesky': variants.append(('raw_c_metric',raw_j,(raw,jnp.asarray(factor.matrix),density)))
    else: emit('raw_c_metric', status='unsupported', notes='This diagnostic only implements Cholesky metric solves')
    target = None
    for name, function, operands in variants:
        jax.block_until_ready(operands)
        t = time.perf_counter(); lowered = jax.jit(function).lower(*operands)
        lower_seconds = time.perf_counter()-t
        t = time.perf_counter(); executable = lowered.compile(); compile_seconds = time.perf_counter()-t
        result, cold = timed(executable, *operands)
        samples = [timed(executable,*operands)[1] for _ in range(3)]
        if target is None: target = np.asarray(result)
        try: stats = executable.memory_analysis()
        except (AttributeError, RuntimeError, NotImplementedError): stats = None
        xla = None if stats is None else {key:getattr(stats,key,None) for key in
            ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
        result_host = np.asarray(result)
        matches = bool(np.allclose(result_host, target, atol=1e-10, rtol=1e-10))
        emit(name, status='ok' if matches else 'numerical_mismatch',
            comparison_atol=1e-10, comparison_rtol=1e-10, trace_lower_seconds=lower_seconds, compile_seconds=compile_seconds,
            cold_seconds=cold, warm_seconds=samples, warm_median_seconds=float(np.median(samples)),
            xla_memory=xla, xla_memory_kind='compiler estimate, not RSS',
            max_abs_j_difference=float(np.max(np.abs(result_host-target))), j_norm=float(np.linalg.norm(result_host)),
            operand_layout=[dict(shape=list(a.shape),dtype=str(a.dtype),bytes=int(a.nbytes)) for a in operands])


if __name__ == '__main__':
    main()
