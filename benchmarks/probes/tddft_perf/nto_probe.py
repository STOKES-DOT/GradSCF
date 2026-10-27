"""Offline NTO analysis of actual Davidson inputs, residuals and states.

debug.callback capture is diagnostic only. Timings from this instrumented
solve are excluded from formal performance comparisons. No approximate action
is passed to the eigensolver.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from worker import Recorder, reference, solver, ready

jax.config.update("jax_enable_x64",True)


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text()); output=Path(sys.argv[2]); output.mkdir(parents=True,exist_ok=True)
    log=Recorder(cfg,output)
    mf,ref=reference(cfg,log)
    import gradscf.tddft.casida as casida
    build=casida.build_restricted_tda_operator
    captured=[]
    call_shapes=[]
    def capture(values):
        call_shapes.append(list(values.shape))
        valid=np.asarray(values)
        valid=valid[np.linalg.norm(valid,axis=1)>1e-8]
        if len(valid) and len(captured)<128:
            captured.append(valid[-1].copy())
    def instrument(*args,**kwargs):
        vind,diag,gaps=build(*args,**kwargs)
        def wrapped(values):
            jax.debug.callback(capture,values)
            return vind(values)
        wrapped.memory_plan=vind.memory_plan
        return wrapped,diag,gaps
    casida.build_restricted_tda_operator=instrument
    td=solver(cfg,mf)
    td.kernel();ready(td.e);jax.effects_barrier()
    casida.build_restricted_tda_operator=build
    from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
    vind,diag,gaps=build(ref,SemilocalResponseFunctional(cfg['xc']),response_kernel_options={'two_electron_mode':'df'})
    no,nv=gaps.shape
    states=np.asarray(td.result.amplitudes)*np.sqrt(2.)
    vectors=[];names=[]
    if captured:
        for label,index in [('early',0),('middle',len(captured)//2),('late',len(captured)-1)]:
            vectors.append(captured[index].reshape(no,nv));names.append('trial-'+label)
    for index in (0,min(3,len(states)-1)):
        state=states[index]
        residual=np.asarray(vind(jnp.asarray(state.reshape(1,-1))))[0].reshape(no,nv)-float(td.e[index])*state
        vectors.extend([state,residual,residual/(np.asarray(diag).reshape(no,nv)-float(td.e[index])+1e-6)])
        names.extend([f'state-{index}',f'residual-{index}',f'preconditioned-{index}'])
    q=jnp.asarray(ref.df_factors)
    c=jnp.asarray(mf.mo_coeff)
    oo=jnp.einsum('Qpq,pi,qj->Qij',q,c[:,:no],c[:,:no]);vv=jnp.einsum('Qpq,pa,qb->Qab',q,c[:,no:],c[:,no:])
    exact=jax.jit(lambda oo,vv,x:jnp.einsum('Qij,Qab,jb->ia',oo,vv,x))
    low=jax.jit(lambda oo,vv,u,v:jnp.einsum('Qir,Qar->ia',jnp.einsum('Qij,jr->Qir',oo,u),jnp.einsum('Qab,br->Qar',vv,v)))
    ready((oo,vv))
    arrays={}
    for name,vector in zip(names,vectors):
        norm=np.linalg.norm(vector)
        if norm<1e-18:continue
        vector=vector/norm
        start=time.perf_counter();u,s,vh=np.linalg.svd(vector,full_matrices=False);svd=time.perf_counter()-start
        x=jnp.asarray(vector);expected=ready(exact(oo,vv,x))
        exact_samples=[]
        for _ in range(10):
            start=time.perf_counter();ready(exact(oo,vv,x));exact_samples.append(time.perf_counter()-start)
        ranks={}
        for tol in (1e-2,1e-4,1e-6):
            rank=next((r for r in range(1,len(s)+1) if np.linalg.norm(s[r:])<=tol*np.linalg.norm(s)),len(s))
            ranks[str(tol)]=rank
        for rank in sorted({1,*ranks.values()}):
            ur=jnp.asarray(u[:,:rank]*s[:rank]);vr=jnp.asarray(vh[:rank].T)
            start=time.perf_counter();approx=ready(low(oo,vv,ur,vr));first=time.perf_counter()-start
            samples=[]
            for _ in range(10):
                start=time.perf_counter();ready(low(oo,vv,ur,vr));samples.append(time.perf_counter()-start)
            error=float(jnp.linalg.norm(approx-expected)/jnp.linalg.norm(expected))
            log.add('nto_offline',float(np.median(samples)),sample=name,rank=rank,ranks_by_x_tolerance=ranks,
                    singular_values=s,svd_seconds=svd,k_exact_samples=exact_samples,k_lowrank_samples=samples,
                    first_lowrank_seconds=first,k_relative_error=error,x_relative_error=float(np.linalg.norm(s[rank:])/np.linalg.norm(s)),
                    notes='offline diagnostic; no approximate eigensolver; SVD/transfer excluded from K-only samples')
        arrays[name]=vector
    np.savez(output/'captured-vectors.npz',**arrays)
    log.add('nto_metadata',None,captured_calls=len(captured),actual_operator_calls=len(call_shapes),
            runtime_action_shapes=call_shapes,converged=bool(td.converged),
            qoo_bytes=oo.size*8,qov_bytes=q.shape[0]*no*nv*8,qvv_bytes=vv.size*8,
            original_operator_residual=[float(np.linalg.norm(np.asarray(vind(jnp.asarray(x.reshape(1,-1))))[0]-float(e)*x.reshape(-1))) for e,x in zip(td.e,states)])


if __name__=='__main__':main()
