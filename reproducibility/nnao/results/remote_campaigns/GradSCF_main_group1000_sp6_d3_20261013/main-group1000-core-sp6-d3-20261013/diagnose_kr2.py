"""Reproduce Kr2 stability and audit the orbital Hessian, not four-index ERIs."""
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.training.objective import _run_scf_with_rescue
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
from gradscf.scf.stability import _df_rhf_orbital_curvature
from gradscf.solvers import LinearOperator, EigenSolverConfig, solve_hermitian

root=Path(__file__).resolve().parent
row=next(r for r in json.loads((root/'inputs/pilot.json').read_text())['structures']
         if r['structure_id']=='GMTKN55-v2/RG18/kr2')
with jax.default_device(jax.devices('cpu')[0]):
    layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                                basis_family='nnao_szp_s',core_primitives='6-31g')
problem,_=prepare_fixed_pool(row,layout,root/'kr2-diagnostic-cache/integrals.npz',storage='mmap')
c=jnp.asarray(layout.reference_outputs())
c=c/jnp.maximum(jnp.linalg.norm(c,axis=-1,keepdims=True),1e-30)
# Inspect the exact fused production executable before separate solver probes.
import gradscf.scf.stability as stability
original_solve=stability.solve_hermitian
def logged_solve(*args,**kwargs):
    out=original_solve(*args,**kwargs)
    jax.debug.print('FUSED values={} residuals={} converged={}',out.values,out.residual_norms,out.converged)
    return out
stability.solve_hermitian=logged_solve
(energy,info),derivative=problem._value_grad(c,*problem.raw)
print('FUSED',json.dumps({k:np.asarray(v).item() for k,v in info.items()}),flush=True)
stability.solve_hermitian=original_solve
s,h,b=problem.project(c,*problem.raw)
n=s.shape[0]; no=layout.nelectron//2; dim=no*(n-no)
cfg=RKSConfig(xc_spec='hf',jk_backend='df',max_cycle=150,
              conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)
kw=dict(overlap=s,hcore=h,df_factors=b,eri=None,nelectron=layout.nelectron,
        nuclear_repulsion=problem.enuc,ao=jnp.zeros((0,n)),
        ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
r,_,_=jax.jit(lambda: _run_scf_with_rescue(run_rks_from_integrals_traceable,kw,cfg,
                                         rescue_level_shift=.2))()
g,apply=_df_rhf_orbital_curvature(h,b,r.mo_coeff,no)
eps=jnp.diag(r.mo_coeff.T@r.fock_matrix@r.mo_coeff)
diag=(4*(eps[no:][None,:]-eps[:no,None])).reshape(-1)
matrix=jax.jit(jax.vmap(apply,in_axes=1,out_axes=1))(jnp.eye(dim))
np.savez(root/'kr2-hessian.npz',hessian=np.asarray(matrix),diagonal=np.asarray(diag))
print(json.dumps(dict(backend=jax.default_backend(),nao=n,dimension=dim,
    converged=bool(r.converged),energy=float(r.total_energy),gradient_norm=float(jnp.linalg.norm(g)),
    asymmetry=float(jnp.linalg.norm(matrix-matrix.T)))),flush=True)
op=LinearOperator((dim,dim),s.dtype,apply,diagonal=diag,
                  matmat=jax.vmap(apply,in_axes=1,out_axes=1))
records=[]
for method,space,iters in [('dense',40,100),('davidson',40,100),('davidson',80,100),('davidson',40,300)]:
    start=time.perf_counter()
    out=jax.jit(lambda: solve_hermitian(op,config=EigenSolverConfig(method=method,nroots=3,
        atol=1e-7,maxiter=iters,max_subspace=min(space,dim),max_dense=512,initial_guess_count=8)))()
    record=dict(method=method,max_subspace=space,maxiter=iters,values=np.asarray(out.values).tolist(),
        residuals=np.asarray(out.residual_norms).tolist(),converged=np.asarray(out.converged).tolist(),
        seconds=time.perf_counter()-start)
    print(json.dumps(record),flush=True); records.append(record)
(root/'kr2-diagnosis.json').write_text(json.dumps(records,indent=2)+'\n')
