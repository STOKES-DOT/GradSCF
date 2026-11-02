"""Exact failed candidate: analytic orbital Hessian check, no finite differences."""
from pathlib import Path
import hashlib
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.model.nnao.training.initial_guess import atomic_guess_data,atomic_density_guess
from gradscf.model.nnao.training.objective import _run_scf_with_rescue
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.runtime import atomic_json,cache_location
from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable
from gradscf.scf.stability import stabilize_df_rhf,_df_rhf_orbital_curvature
from gradscf.solvers import LinearOperator,EigenSolverConfig,solve_hermitian

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'inputs/references.json').read_text())
row=next(r for r in data['structures'] if r['structure_id']=='GMTKN55-v2/RG18/kr3')
pool=json.loads((ROOT/'selection/candidate.json').read_text())
model=json.loads((ROOT/'inputs/model-config.json').read_text())
with np.load(ROOT/'train/failed-update-000030.npz',allow_pickle=False) as a:
    parameters=a['parameters'];metadata=json.loads(a['metadata'].item())
assert metadata['accepted_updates']==29
_,predict,_,_=make_mace_predictor(0,model,return_mask=True)
with jax.default_device(jax.devices('cpu')[0]):
    layout=bind_pool_candidate(prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
        basis_family=data['basis_family'],core_primitives=data['core_primitives']),pool)
free,_=prepare_fixed_pool(row,layout,cache_location(ROOT/'cache',row),storage='device')
graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=model['elements'],cutoff=model['cutoff'])
c=predict(jnp.asarray(parameters),graph);s,h,b=jax.block_until_ready(free.project(c,*free.raw))
n=s.shape[0];no=layout.nelectron//2;dimension=no*(n-no)
cfg=RKSConfig(xc_spec='hf',jk_backend='df',max_cycle=150,conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)
mask,occ=atomic_guess_data(layout)
def inputs(s,h,b):
    return dict(overlap=s,hcore=h,eri=None,df_factors=b,nelectron=layout.nelectron,nuclear_repulsion=free.enuc,
        ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
@jax.jit
def raw(s,h,b):
    return _run_scf_with_rescue(run_rks_from_integrals_traceable,inputs(s,h,b),cfg,
        rescue_level_shift=.2,fallback_guess=lambda:atomic_density_guess(s,jnp.asarray(mask),jnp.asarray(occ)))[0]
result=jax.block_until_ready(raw(s,h,b))
@jax.jit
def original(result,s,h,b):
    return stabilize_df_rhf(result,**inputs(s,h,b),config=cfg)[1]
old=jax.block_until_ready(original(result,s,h,b))
report=dict(structure_id=row['structure_id'],parameter_sha256=hashlib.sha256(parameters.tobytes()).hexdigest(),
    device=jax.devices('gpu')[0].device_kind,nao=n,nocc=no,dimension=dimension,
    energy_hartree=float(result.total_energy),converged=bool(result.converged),
    original={k:np.asarray(v).item() for k,v in old.items()},finite_differences=False,status='running')
atomic_json(ROOT/'kr3-diagnostic.json',report);print(report,flush=True)
@jax.jit
def dense(h,b,coeff):
    g,apply=_df_rhf_orbital_curvature(h,b,coeff,no)
    # Column-wise construction bounds temporary storage; this is an orbital
    # Hessian, never a four-index ERI tensor.
    operator=LinearOperator((dimension,dimension),h.dtype,apply)
    solved=solve_hermitian(operator,config=EigenSolverConfig(method='dense',nroots=3,atol=1e-7,max_dense=1024))
    return solved,jnp.linalg.norm(g)
tick=time.perf_counter();solved,gradient=jax.block_until_ready(dense(h,b,result.mo_coeff))
report.update(status='complete',dense=dict(values=np.asarray(solved.values).tolist(),
    residuals=np.asarray(solved.residual_norms).tolist(),converged=np.asarray(solved.converged).tolist(),
    gradient_norm=float(gradient),seconds=time.perf_counter()-tick))
atomic_json(ROOT/'kr3-diagnostic.json',report);print(report['dense'],flush=True)
