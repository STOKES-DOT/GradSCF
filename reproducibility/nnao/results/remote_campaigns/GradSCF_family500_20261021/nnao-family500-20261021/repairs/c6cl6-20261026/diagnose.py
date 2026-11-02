"""Compare AO and invertibly whitened SCF on the exact failed initial basis."""
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from gradscf.data.molecule import atomic_number
from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, cache_location
from gradscf.model.nnao.training.initial_guess import atomic_density_guess, atomic_guess_data
from gradscf.model.nnao.training.objective import _run_scf_with_rescue
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.scf.core import _orthogonalizer, _restricted_density_from_fock
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable, _mo_residual_norm


ROOT = Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021')
HERE = ROOT/'repairs/c6cl6-20261026'


def main():
    started=time.perf_counter()
    row=next(r for r in json.loads((ROOT/'inputs/references.json').read_text())['structures']
             if r['structure_id']=='GMTKN55-v2/DC13/C6Cl6')
    candidate=json.loads((ROOT/'inputs/l-candidate.json').read_text())
    model=dict(json.loads((ROOT/'inputs/model-config.json').read_text()),basis_family='nnao_tzp_l')
    with jax.default_device(jax.devices('cpu')[0]):
        layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                                   basis_family='nnao_tzp_l',core_primitives='6-31g')
        layout=bind_pool_candidate(layout,candidate)
    problem,manifest=prepare_fixed_pool(row,layout,cache_location(ROOT/'cache/l',row),storage='mmap')
    initial,predict,_=make_mace_predictor(0,model)
    with np.load(ROOT/'runs/nnao_tzp_l/train/failed-update-000001.npz',allow_pickle=False) as saved:
        parameters=jnp.asarray(saved['parameters'])
    # CPU/GPU initialization differs by a few ulps; always use the saved vector.
    np.testing.assert_allclose(parameters,initial,atol=5e-14,rtol=5e-14)
    graph=build_graph([atomic_number(s) for s in row['symbols']],row['coords_angstrom'],
                      element_order=model['elements'],cutoff=model.get('cutoff',5.))
    coefficients=predict(parameters,graph)
    inputs=jax.block_until_ready(problem.project(coefficients,*problem.raw))
    print(json.dumps(dict(event='prepared',nao=layout.topology.nao,
                          elapsed_seconds=time.perf_counter()-started)),flush=True)
    atomic=atomic_guess_data(layout)
    cfg=RKSConfig(xc_spec='hf',jk_backend='df',max_cycle=150,
                  conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)

    def evaluate(physical,whitened):
        s,h,b=physical;n=s.shape[0]
        lower=jnp.linalg.cholesky(s)
        x=solve_triangular(lower.T,jnp.eye(n,dtype=s.dtype),lower=False)
        seed=atomic_density_guess(s,*atomic)
        if whitened:
            h=x.T@h@x;b=x.T@b@x;s=jnp.eye(n,dtype=s.dtype)
            seed=lower.T@seed@lower
        kwargs=dict(overlap=s,hcore=h,df_factors=b,eri=None,
            nelectron=layout.nelectron,nuclear_repulsion=problem.enuc,
            ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
        result,rescued,cycles=_run_scf_with_rescue(run_rks_from_integrals_traceable,
            kwargs,cfg,rescue_level_shift=.2,fallback_guess=lambda:seed)
        density=result.density_matrix
        j,k=build_jk_from_df(b,density);f=h+j-.5*k
        next_density=_restricted_density_from_fock(f,_orthogonalizer(s,1e-10),layout.nelectron//2)
        jnext,knext=build_jk_from_df(b,next_density)
        next_energy=jnp.sum(next_density*h)+.5*jnp.sum(next_density*jnext)-.25*jnp.sum(next_density*knext)+problem.enuc
        density_ao=x@density@x.T if whitened else density
        delta_ao=x@(next_density-density)@x.T if whitened else next_density-density
        delta_orth=next_density-density if whitened else lower.T@delta_ao@lower
        return density_ao,dict(converged=result.converged,cycles=cycles,rescued=rescued,
            energy=result.total_energy,refill_energy_change=next_energy-result.total_energy,
            fixed_point_ao=jnp.linalg.norm(delta_ao),fixed_point_orthonormal=jnp.linalg.norm(delta_orth),
            orbital_gradient=_mo_residual_norm(f,result.mo_coeff,result.mo_occ),
            commutator=jnp.linalg.norm(f@density@s-s@density@f),
            electron_count=jnp.trace(density@s),min_overlap=jnp.linalg.eigvalsh(physical[0])[0],
            max_overlap=jnp.linalg.eigvalsh(physical[0])[-1])

    report=dict(structure_id=row['structure_id'],job_id=os.environ.get('SLURM_JOB_ID'),
        devices=[d.device_kind for d in jax.devices()],jax=jax.__version__,
        candidate_sha256=manifest['cache_sha256'],same_seed_parameters=True,
        nao=layout.topology.nao,nelectron=layout.nelectron,finite_differences=False)
    densities=[]
    for whitened,name in ((False,'ao'),(True,'orthonormal')):
        begin=time.perf_counter()
        density,info=jax.block_until_ready(jax.jit(lambda p:evaluate(p,whitened))(inputs))
        report[name]={key:np.asarray(value).item() for key,value in info.items()}
        report[name]['elapsed_seconds']=time.perf_counter()-begin
        densities.append(density)
        atomic_json(HERE/'diagnostics.json',report)
        print(json.dumps(dict(event=name,**report[name])),flush=True)
    lower=jnp.linalg.cholesky(inputs[0])
    report['between_solutions_orthonormal_density_difference']=float(jnp.linalg.norm(lower.T@(densities[0]-densities[1])@lower))
    report['elapsed_seconds']=time.perf_counter()-started
    atomic_json(HERE/'diagnostics.json',report)


if __name__=='__main__':main()
