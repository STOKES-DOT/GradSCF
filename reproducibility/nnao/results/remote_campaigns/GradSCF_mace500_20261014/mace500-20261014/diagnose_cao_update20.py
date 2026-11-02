"""Reproduce the exact accepted update-20 CaO SCF and test bounded seeds."""
from dataclasses import replace
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
from gradscf.model.nnao.training.initial_guess import atomic_guess_data, atomic_density_guess
from gradscf.model.nnao.training.objective import _run_scf_with_rescue
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.runtime import atomic_json, cache_location
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
from gradscf.scf.stability import stabilize_df_rhf
from gradscf.integrals.molecular.jk import build_jk_from_df

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'inputs/references.json').read_text())
row = next(r for r in data['structures'] if r['structure_id']=='GMTKN55-v2/ALKBDE10/cao')
candidate = json.loads((ROOT/'selection/candidate.json').read_text())
model = json.loads((ROOT/'inputs/model-config.json').read_text())
initial, predict, _, _ = make_mace_predictor(0, model, return_mask=True)
with np.load(ROOT/'train/last.npz', allow_pickle=False) as archive:
    parameters = archive['parameters']
    metadata = json.loads(archive['metadata'].item())
assert metadata['update_step']==20 and metadata['batch_cursor']==10
with jax.default_device(jax.devices('cpu')[0]):
    layout = bind_pool_candidate(prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
        basis_family=data['basis_family'],core_primitives=data['core_primitives']), candidate)
free, _ = prepare_fixed_pool(row,layout,cache_location(ROOT/'cache',row),storage='device')
graph = build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=model['elements'],cutoff=model['cutoff'])
coefficients = predict(jnp.asarray(parameters),graph)
s,h,b = jax.block_until_ready(free.project(coefficients,*free.raw))
n = s.shape[0]
cfg = RKSConfig(xc_spec='hf',jk_backend='df',max_cycle=150,
    conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)
mask,occ = atomic_guess_data(layout)
guess = atomic_density_guess(s,jnp.asarray(mask),jnp.asarray(occ))
def kwargs(s,h,b):
    return dict(overlap=s,hcore=h,eri=None,df_factors=b,nelectron=layout.nelectron,
        nuclear_repulsion=free.enuc,ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
report=dict(status='running',structure_id=row['structure_id'],update_step=20,
    parameter_sha256=hashlib.sha256(parameters.tobytes()).hexdigest(),nao=n,
    device=jax.devices('gpu')[0].device_kind,results=[])
np.savez(ROOT/'cao-update20-physical.npz',overlap=np.asarray(s),hcore=np.asarray(h),
    df_factors=np.asarray(b),atomic_guess=np.asarray(guess),coefficients=np.asarray(coefficients),
    nelectron=layout.nelectron,enuc=float(free.enuc))
def record(label,result,seconds):
    j,k=build_jk_from_df(b,result.density_matrix);f=h+j-.5*k
    entry=dict(label=label,converged=bool(result.converged),energy_hartree=float(result.total_energy),
        cycles=int(result.cycles),orbital_residual=float(jnp.linalg.norm(f@result.density_matrix@s-s@result.density_matrix@f)),
        seconds=seconds)
    if entry['converged']:
        stable,info=stabilize_df_rhf(result,**kwargs(s,h,b),config=cfg)
        entry.update({k:float(v) if k not in ['stable','stability_resolved'] else bool(v) for k,v in info.items()})
        entry['stable_energy_hartree']=float(stable.total_energy)
    report['results'].append(entry);atomic_json(ROOT/'cao-update20-diagnostic.json',report);print(entry,flush=True)
    return entry
@jax.jit
def baseline(s,h,b,guess):
    return _run_scf_with_rescue(run_rks_from_integrals_traceable,kwargs(s,h,b),cfg,
        rescue_level_shift=.2,fallback_guess=lambda:guess)[0]
tick=time.perf_counter();result=jax.block_until_ready(baseline(s,h,b,guess))
record('unchanged_rescue',result,time.perf_counter()-tick)
for shift,damping in [(.2,.0),(.5,.2),(1.,.2),(2.,.2)]:
    def solve(s,h,b,guess):
        shifted=run_rks_from_integrals_traceable(**dict(kwargs(s,h,b),init_density=guess),
            config=replace(cfg,level_shift=shift,damping=damping))
        polished=run_rks_from_integrals_traceable(**dict(kwargs(s,h,b),init_density=shifted.density_matrix,
            init_mo_coeff=shifted.mo_coeff,init_mo_energy=shifted.mo_energy),config=cfg)
        return shifted,polished
    tick=time.perf_counter();shifted,result=jax.block_until_ready(jax.jit(solve)(s,h,b,guess))
    entry=record(f'atomic_shift_{shift}_damping_{damping}_raw_polish',result,time.perf_counter()-tick)
    entry.update(shifted_converged=bool(shifted.converged),shifted_cycles=int(shifted.cycles))
    atomic_json(ROOT/'cao-update20-diagnostic.json',report)
report['status']='complete';atomic_json(ROOT/'cao-update20-diagnostic.json',report)
