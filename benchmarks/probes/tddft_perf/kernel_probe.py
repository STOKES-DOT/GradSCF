"""Shared-input packed J/K and DF transform diagnostics, including AD.

Synthetic kernels are not molecular end-to-end timings. Dynamic ERIs/factors
prevent constant folding. Exact and rearranged actions receive identical data.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from worker import Recorder, ready

jax.config.update("jax_enable_x64", True)


def measure(name, fn, arguments, log):
    start = time.perf_counter()
    lowered = jax.jit(fn).lower(*arguments)
    log.add(name+"_lower", time.perf_counter()-start)
    start = time.perf_counter()
    compiled = lowered.compile()
    memory = compiled.memory_analysis()
    attrs = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")
    log.add(name+"_compile", time.perf_counter()-start, compiler_memory={a:getattr(memory,a,None) for a in attrs})
    (log.output/(name+".hlo.txt")).write_text(compiled.as_text())
    for repeat in range(11):
        out = log.timed(name+"_first" if repeat == 0 else name+"_warm", lambda: compiled(*arguments), repeat)
    return out


def packed(cfg, log):
    from gradscf.tddft.response import _jk_from_eri_pair_matrix
    from gradscf.integrals.molecular.jk import build_jk_from_packed
    n = cfg["nao"]
    log.context.update(nao=n, batch_size=5)
    rng = np.random.default_rng(7)
    size = n*(n+1)//2
    raw = rng.normal(size=(size, 12))*.01
    pair = jnp.asarray(raw@raw.T)
    density = jnp.asarray(rng.normal(size=(5, n, n)))
    fn = _jk_from_eri_pair_matrix if cfg.get("experiment") != "packed-public" else build_jk_from_packed
    out = measure("packed_jk", fn, (pair, density), log)
    if cfg.get('timing_only'):
        log.add('validation_scope',None,notes='clean memory run; numerical/AD oracle executed in separate corresponding validated process')
        return
    oracle = _jk_from_eri_pair_matrix(pair, density)
    error = max(float(jnp.max(abs(a-b))) for a,b in zip(out, oracle))
    log.add("correctness", None, gradient_error=None, action_max_abs_error=error,
            notes="real s4 pair symmetry; arbitrary non-symmetric batched transition density")
    assert error < 1e-10
    if n == 25:
        dp = jnp.cos(jnp.arange(pair.size).reshape(pair.shape))*.001
        dd = jnp.sin(jnp.arange(density.size).reshape(density.shape))*.01
        cot = tuple(jnp.ones_like(a)*.01 for a in out)
        def product(a, b):
            return sum(jnp.vdot(v,w) for v,w in zip(fn(a,b), cot))
        _, tangent = log.timed("jk_jvp", lambda: jax.jvp(fn,(pair,density),(dp,dd)))
        derivative = log.timed("jk_vjp", lambda: jax.grad(product,argnums=(0,1))(pair,density))
        lhs = sum(jnp.vdot(a,b) for a,b in zip(tangent,cot))
        rhs = jnp.vdot(derivative[0],dp)+jnp.vdot(derivative[1],dd)
        h=1e-4
        fd=(product(pair+h*dp,density+h*dd)-product(pair-h*dp,density-h*dd))/(2*h)
        log.add("jk_ad_check", None, gradient_error=float(abs(lhs-rhs)), finite_difference_error=float(abs(fd-lhs)))
        assert float(abs(lhs-rhs)) < 1e-10 and float(abs(fd-lhs)) < 1e-7


def lowrank(cfg, log):
    from gradscf.tddft.lowrank_response import build_restricted_lowrank_mo_response_action as original
    from experiments import lowrank_factory
    rng = np.random.default_rng(7)
    n, no, nq = 60, 11, 150
    # Deliberately NON-symmetric, unequal occupied/virtual dimensions.
    q = jnp.asarray(rng.normal(size=(nq,n,n))*.01)
    co = jnp.asarray(np.linalg.qr(rng.normal(size=(n,n)))[0])
    x = jnp.asarray(rng.normal(size=(5,no,n-no)))
    y = jnp.asarray(rng.normal(size=x.shape))
    log.context.update(nao=n,nocc=no,nvir=n-no,naux=nq,batch_size=5)
    factory = lowrank_factory(full=cfg["full"]) if cfg.get("experiment") in {"df-demand", "df-share"} else original
    def function(q, co, x, y):
        act = factory(q,q,co[:,:no],co[:,no:],.25,include_exchange=True,dtype=q.dtype)
        out=act(x,bottom_density=False,bottom_projection=False)
        if cfg["full"]:
            upper=out+act(y,bottom_density=True,bottom_projection=False)
            lower=act(x,bottom_density=False,bottom_projection=True)+act(y,bottom_density=True,bottom_projection=True)
            return upper,lower
        return (out,)
    def oracle(q, co, x, y):
        act = original(q,q,co[:,:no],co[:,no:],.25,include_exchange=True,dtype=q.dtype)
        if cfg["full"]:
            return (act(x,bottom_density=False,bottom_projection=False)+act(y,bottom_density=True,bottom_projection=False),
                    act(x,bottom_density=False,bottom_projection=True)+act(y,bottom_density=True,bottom_projection=True))
        return (act(x,bottom_density=False,bottom_projection=False),)
    out = measure("df_outer_jit",function,(q,co,x,y),log)
    expected=oracle(q,co,x,y)
    error=max(float(jnp.max(abs(a-b))) for a,b in zip(out,expected))
    log.add("correctness",None,action_max_abs_error=error)
    assert error<1e-10
    # Distinct J/K tensors and all required orientations retain the public
    # raw-factor contract; no transpose-symmetry assumption is hidden here.
    other=q*.7+jnp.sin(q)
    left=factory(q,other,co[:,:no],co[:,no:],.25,include_exchange=True,dtype=q.dtype)
    right=original(q,other,co[:,:no],co[:,no:],.25,include_exchange=True,dtype=q.dtype)
    orientations=[(False,False)] if not cfg["full"] else [(a,b) for a in (False,True) for b in (False,True)]
    difference=max(float(jnp.max(abs(left(x,bottom_density=a,bottom_projection=b)-right(x,bottom_density=a,bottom_projection=b)))) for a,b in orientations)
    log.add("distinct_jk_factors",None,action_max_abs_error=difference)
    assert difference<1e-10
    for repeat in range(4):
        log.timed("df_eager_prepare",lambda:factory(q,q,co[:,:no],co[:,no:],.25,include_exchange=True,dtype=q.dtype),repeat)
    direction=jnp.cos(jnp.arange(q.size).reshape(q.shape))*.001
    scalar=lambda q:sum(jnp.sum(a**2) for a in function(q,co,x,y))
    oldscalar=lambda q:sum(jnp.sum(a**2) for a in oracle(q,co,x,y))
    derivative=log.timed("factor_vjp",lambda:jax.jit(jax.grad(scalar))(q))
    expected=log.timed("factor_vjp_oracle",lambda:jax.jit(jax.grad(oldscalar))(q))
    log.add("factor_ad_check",None,gradient_error=float(jnp.max(abs(derivative-expected))))
    assert float(jnp.max(abs(derivative-expected)))<1e-10


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text()); output=Path(sys.argv[2]); output.mkdir(parents=True,exist_ok=True)
    log=Recorder(cfg,output)
    (packed if cfg.get("probe")=="packed" else lowrank)(cfg,log)
    if cfg.get("device")=="gpu":
        stats=jax.devices('gpu')[0].memory_stats()
        log.add('gpu_allocator',None,gpu_memory_bytes=stats.get('peak_bytes_in_use') if stats else None,
                gpu_memory_metric_kind='JAX allocator lifetime peak bytes in use; not reserved VRAM',allocator_stats=stats)


if __name__ == "__main__":
    main()
