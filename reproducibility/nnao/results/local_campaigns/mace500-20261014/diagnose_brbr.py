"""Isolate the failed initial model using unchanged cached integrals and tolerances."""
from dataclasses import replace
from pathlib import Path
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.runtime import cache_location,atomic_json
from gradscf.model.nnao.training.objective import _run_scf_with_rescue
from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable
from gradscf.scf.stability import stabilize_df_rhf
from gradscf.integrals.molecular.jk import build_jk_from_df

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'inputs/references.json').read_text())
row=next(r for r in data['structures'] if r['structure_id']=='GMTKN55-v2/HAL59/BrBr_OPH3')
candidate=json.loads((ROOT/'selection/candidate.json').read_text())
model_config=json.loads((ROOT/'inputs/model-config.json').read_text())
cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
with jax.default_device(cpu):
    layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),basis_family=data['basis_family'],core_primitives=data['core_primitives'])
    layout=bind_pool_candidate(layout,candidate)
free,_=prepare_fixed_pool(row,layout,cache_location(ROOT/'cache',row),storage='device')
initial,predict,_,_=make_mace_predictor(0,model_config,return_mask=True)
graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=model_config['elements'])
coeff=predict(initial,graph)
s,h,b=jax.block_until_ready(free.project(coeff,*free.raw));n=s.shape[0]
kwargs=dict(overlap=s,hcore=h,eri=None,df_factors=b,nelectron=layout.nelectron,nuclear_repulsion=free.enuc,
    ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
cfg=RKSConfig(xc_spec='hf',jk_backend='df',max_cycle=150,conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)
report=dict(structure_id=row['structure_id'],nao=n,nelectron=layout.nelectron,device=gpu.device_kind,results=[])
def record(label,result,seconds):
    j,k=build_jk_from_df(b,result.density_matrix);f=h+j-.5*k
    entry=dict(label=label,converged=bool(result.converged),energy_hartree=float(result.total_energy),cycles=int(result.cycles),
        orbital_residual=float(jnp.linalg.norm(f@result.density_matrix@s-s@result.density_matrix@f)),seconds=seconds)
    if entry['converged']:
        stable,info=stabilize_df_rhf(result,**kwargs,config=cfg)
        entry.update({k:float(v) if k not in ['stable','stability_resolved'] else bool(v) for k,v in info.items()})
        entry['stable_energy_hartree']=float(stable.total_energy)
    report['results'].append(entry);atomic_json(ROOT/'brbr-diagnostic.json',report);print(entry,flush=True)

tick=time.perf_counter()
baseline=jax.jit(lambda:_run_scf_with_rescue(run_rks_from_integrals_traceable,kwargs,cfg,rescue_level_shift=.2))()[0]
jax.block_until_ready(baseline);record('unchanged_rescue',baseline,time.perf_counter()-tick)

# A neutral-atom population guess; no isolated-atom SCF or PySCF calls.
occupations=np.zeros(n);groups=[];offset=0
for atom,z in enumerate(row['atomic_numbers']):
    begin=offset;core=0;valence=[]
    for owner,l,role,nc in zip(layout.shell_atoms,layout.topology.angular_momenta,layout.roles,layout.topology.contraction_counts):
        if owner!=atom:continue
        width=(2*l+1)*nc
        if role=='core':occupations[offset:offset+width]=2.;core+=2*width
        if role.startswith('valence_'):valence.append((offset,l,nc))
        offset+=width
    remaining=z-core
    for start,l,nc in sorted(valence,key=lambda x:x[1]):
        count=min(remaining,2*(2*l+1));occupations[start:start+2*l+1]=count/(2*l+1);remaining-=count
    assert remaining==0;groups.append((begin,offset))
sm=np.asarray(s);guess=np.zeros_like(sm)
for begin,end in groups:
    e,u=np.linalg.eigh(sm[begin:end,begin:end]);x=(u/np.sqrt(e))@u.T
    guess[begin:end,begin:end]=(x*occupations[begin:end])@x.T
np.testing.assert_allclose(np.trace(guess@sm),layout.nelectron,atol=1e-10)
guess=jax.device_put(guess,gpu)
tick=time.perf_counter()
atomic=jax.jit(lambda:run_rks_from_integrals_traceable(**dict(kwargs,init_density=guess),config=cfg))()
jax.block_until_ready(atomic);record('neutral_atomic_population',atomic,time.perf_counter()-tick)
for shift in [1.,2.]:
    tick=time.perf_counter()
    shifted=jax.jit(lambda:run_rks_from_integrals_traceable(**kwargs,config=replace(cfg,level_shift=shift,max_cycle=300)))()
    polished=jax.jit(lambda:run_rks_from_integrals_traceable(**dict(kwargs,init_density=shifted.density_matrix,
        init_mo_coeff=shifted.mo_coeff,init_mo_energy=shifted.mo_energy),config=cfg))()
    jax.block_until_ready(polished);record(f'shift_{shift}_raw_polish',polished,time.perf_counter()-tick)
report['status']='complete';atomic_json(ROOT/'brbr-diagnostic.json',report)
