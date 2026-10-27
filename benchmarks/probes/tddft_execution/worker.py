"""Production-only before/after solves; diagnostics and clean phases separate.

Per-solve deadlines are reported to the parent independently of its aggregate
process budget. Original operator certification runs after the clean-phase
memory marker and cannot inflate that phase's peak metric.
"""
from __future__ import annotations
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('td_probe_parent_worker',ROOT/'benchmarks/probes/tddft_perf/worker.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
Recorder,ready,reference,solver,plain=(getattr(parent,name) for name in ('Recorder','ready','reference','solver','plain'))


def original_residuals(td,full):
    result=td.result
    raw=td._solver
    if full:
        vind=raw.gen_tdhf_vind()
        x=np.asarray(result.x_amplitudes).reshape(len(td.e),-1)
        y=np.asarray(result.y_amplitudes).reshape(len(td.e),-1)
        vectors=np.concatenate((x,y),axis=1)
    else:
        vind=raw.gen_tda_vind()
        if hasattr(result,'amplitudes'):
            vectors=np.asarray(result.amplitudes).reshape(len(td.e),-1)
        else:
            vectors=np.concatenate((np.asarray(result.amplitudes_alpha).reshape(len(td.e),-1),
                                    np.asarray(result.amplitudes_beta).reshape(len(td.e),-1)),axis=1)
    acted=np.asarray(ready(vind(vectors)))
    norms=np.linalg.norm(vectors,axis=1)
    return np.linalg.norm(acted-np.asarray(td.e)[:,None]*vectors,axis=1)/norms


def compile_counter(output,repeat):
    from jax._src import compiler
    import traceback
    original=compiler.backend_compile_and_load
    def record(*args,**kwargs):
        module=args[1] if len(args)>1 else kwargs.get('module')
        info={}
        try:
            info['module_name']=str(module.operation.attributes['sym_name'])
            info['functions']=[dict(name=str(op.attributes['sym_name']),signature=str(op.attributes['function_type']))
                for op in module.body.operations if op.operation.name=='func.func']
        except (AttributeError,KeyError,TypeError):info['module_repr']=repr(module)
        info.update(repeat=repeat,source='backend_compile_and_load; diagnostic only',
                    stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_stack(limit=22)])
        started=time.perf_counter()
        try:return original(*args,**kwargs)
        finally:
            info['wall_seconds']=time.perf_counter()-started
            with (output/'compile_events.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
    compiler.backend_compile_and_load=record
    return lambda:setattr(compiler,'backend_compile_and_load',original)


def workload_counter(output):
    from gradscf.tddft import tda,casida,unrestricted
    from gradscf.solvers.eigen import api
    from gradscf.solvers.eigen._workload import _workload_reporter
    import jax.numpy as jnp
    def deliver(event):
        event=dict(event,source='opt-in solver workload; scalar host callbacks',
                   action_device_seconds=None,
                   timing_caveat='host callback interval includes instrumentation; device scheduling may overlap; not a clean kernel time')
        with (output/'solver_events.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(event),allow_nan=False)+'\n')
    for module,name in [(tda,'solve_hermitian'),(unrestricted,'solve_hermitian'),
                        (casida,'implicit_differential_davidson_lowest_tdhf'),
                        (unrestricted,'implicit_differential_davidson_lowest_tdhf')]:
        original=getattr(module,name)
        def wrapped(*args,_original=original,**kwargs):
            return _original(*args,workload_callback=deliver,**kwargs)
        setattr(module,name,wrapped)
    # This action follows solve_ritz, outside Davidson's own accounting. Keep
    # it in the diagnostic trace rather than silently treating that trace as
    # all operator work. Derivative diagnostics, if requested, are also tagged.
    original_spectral=api.spectral_values
    def spectral(apply,basis,**options):
        scope=_workload_reporter(deliver,solver='spectral_observable',
            nroots=basis.shape[1],max_subspace=basis.shape[1],tolerance=jnp.asarray(-1.))
        def observed(vectors):
            return scope.apply(apply,vectors,iteration=None,purpose='observable_or_response',
                effective_columns=jnp.count_nonzero(jnp.linalg.norm(vectors,axis=0)>1e-12),
                subspace_dim=basis.shape[1],residual_norms=jnp.full((basis.shape[1],),jnp.nan))
        return original_spectral(observed,basis,**options)
    api.spectral_values=spectral


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=args.output
    out.mkdir(parents=True,exist_ok=True)
    import jax
    jax.config.update('jax_enable_x64',True)
    if cfg.get('compile_diagnostic'):
        jax.config.update('jax_log_compiles',True)
        jax.config.update('jax_explain_cache_misses',True)
    log=Recorder(cfg,out)
    if cfg.get('solver_diagnostic'):
        workload_counter(out)
    td=None;records=[]
    for repeat in range(cfg.get('repeats',4)):
        started=time.perf_counter()
        mf,ref=reference(cfg,log,repeat)
        td=solver(cfg,mf)
        active=dict(stage='response_solve',repeat=repeat,started_monotonic=time.monotonic(),
                    solve_budget_seconds=cfg.get('solve_budget_seconds',600))
        (out/'active_stage.json').write_text(json.dumps(active)+'\n')
        restore=compile_counter(out,repeat) if cfg.get('compile_diagnostic') else lambda:None
        try:log.timed('response_solve',td.kernel,repeat)
        finally:restore()
        (out/'active_stage.json').write_text(json.dumps(dict(stage='properties',repeat=repeat))+'\n')
        strengths=log.timed('oscillator_strength',td.oscillator_strength,repeat)
        log.add('end_to_end',time.perf_counter()-started,repeat,
                excitations_ha=np.asarray(td.e),oscillator_strength=strengths,
                public_converged=bool(np.all(td.converged)),
                notes='independent geometry->SCF->response->strengths; residual certification follows outside timing')
        records=log.records
    clean=dict(ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               ended_monotonic=time.monotonic(),metric='Linux process high-water mark before certification')
    (out/'clean_phase_done.json').write_text(json.dumps(clean)+'\n')
    (out/'active_stage.json').write_text(json.dumps(dict(stage='original_operator_certificate'))+'\n')
    if td is not None:
        payload=dict(mo_coeff=np.asarray(ref.mo_coeff),mo_occ=np.asarray(ref.mo_occ),
                     overlap=np.asarray(ref.overlap_matrix),energies=np.asarray(td.e))
        for name in ('mo_energy','rdm1','fock_matrix'):
            if getattr(ref,name,None) is not None:payload[name]=np.asarray(getattr(ref,name))
        for name in ('amplitudes','x_amplitudes','y_amplitudes','amplitudes_alpha','amplitudes_beta',
                     'x_amplitudes_alpha','x_amplitudes_beta','y_amplitudes_alpha','y_amplitudes_beta'):
            if hasattr(td.result,name):payload[name]=np.asarray(getattr(td.result,name))
        np.savez(out/'states.npz',**payload)
        residuals=log.timed('original_operator_certificate',lambda:original_residuals(td,cfg['full']))
        log.add('certification',None,residual_per_root=residuals,
                response_converged_per_root=residuals<=cfg['td_tol'],
                scientific_pass=bool(np.all(residuals<=cfg['td_tol'])),
                notes='norm(Az-omega*z)/norm(z), original operator; no dense matrix')
    import gradscf
    from gradscf.integrals._native import register_integrals
    lib=Path(register_integrals()._name)
    runtime=dict(gradscf=gradscf.__file__,native=str(lib),native_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
                 python=sys.executable,slurm_job=os.environ.get('SLURM_JOB_ID'),
                 devices=[dict(name=str(d),kind=d.device_kind,stats=d.memory_stats()) for d in jax.devices()],
                 affinity=sorted(os.sched_getaffinity(0)))
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
