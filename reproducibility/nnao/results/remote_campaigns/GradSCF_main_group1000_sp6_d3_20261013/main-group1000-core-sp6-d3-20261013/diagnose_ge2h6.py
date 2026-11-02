"""Audit the failing S-DZ contraction backward on the actual pilot geometry."""
import importlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool

root=Path(__file__).resolve().parent
row=next(r for r in json.loads((root/'inputs/pilot.json').read_text())['structures']
         if r['structure_id']=='GMTKN55-v2/HEAVYSB11/ge2h6')
with jax.default_device(jax.devices('cpu')[0]):
    layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                                basis_family='nnao_dzp_s',core_primitives='6-31g')
problem,_=prepare_fixed_pool(row,layout,root/'ge2h6-diagnostic-cache/integrals.npz',storage='mmap')
c=jnp.asarray(layout.reference_outputs())
c=c/jnp.maximum(jnp.linalg.norm(c,axis=-1,keepdims=True),1e-30)
linear=importlib.import_module('gradscf.solvers.linear.implicit')
original=linear.linear_residual
def audit(operator, solution, rhs, **kwargs):
    norm,valid=original(operator,solution,rhs,**kwargs)
    jax.debug.print('ADJOINT norm={} rhs_norm={} valid={} finite_solution={} primal_valid={}',
        norm,jnp.linalg.norm(rhs),valid,jnp.all(jnp.isfinite(solution)),kwargs.get('converged',True))
    return norm,valid
linear.linear_residual=audit
gmres_module=importlib.import_module('gradscf.solvers.linear.gmres')
upstream_gmres=gmres_module.gmres
def audit_gmres(operator,rhs,**kwargs):
    result,status=upstream_gmres(operator,rhs,**kwargs)
    jax.debug.print('GMRES '+kwargs['solve_method']+' tol={} norm={} rhs={} finite={}',
        kwargs['tol'],jnp.linalg.norm(operator(result)-rhs),jnp.linalg.norm(rhs),jnp.all(jnp.isfinite(result)))
    return result,status
gmres_module.gmres=audit_gmres
(energy,info),gradient=problem._value_grad(c,*problem.raw)
report={'backend':jax.default_backend(),'energy':float(energy),
        'info':{k:np.asarray(v).item() for k,v in info.items()},
        'finite_gradient':bool(jnp.all(jnp.isfinite(gradient))),
        'finite_entries':int(jnp.sum(jnp.isfinite(gradient))),'gradient_size':int(gradient.size)}
print('FUSED',json.dumps(report),flush=True)
physical=problem.project(c,*problem.raw)
(value,details),cotangent=problem.gradient(physical,problem.enuc)
print('PHYSICAL',json.dumps([dict(shape=list(g.shape),finite=bool(jnp.all(jnp.isfinite(g)))) for g in cotangent]),flush=True)
dummy=tuple(jnp.ones_like(v)*1e-3 for v in physical)
project_gradient=jax.jit(lambda x:jax.vjp(lambda z:problem.project(z,*problem.raw),x)[1](dummy)[0])(c)
print('PROJECTION',bool(jnp.all(jnp.isfinite(project_gradient))),flush=True)
(root/'ge2h6-diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
