"""Frozen-model DF-RHF inference on the historical panel; no gradients or fitting."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import integrals,scf
from gradscf.integrals._native import register_integrals
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.runtime import atomic_json,file_hash,scientific_source_hash,repository_root
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.model.nnao.training.objective import _run_scf_with_rescue,BasisRHF
from gradscf.model.nnao.training.initial_guess import atomic_guess_data,atomic_density_guess
from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable
from gradscf.scf.stability import stabilize_df_rhf
from gradscf.scf.core import _build_density_from_occ,_diagonalize_fock,_orthogonalizer

OUT=Path(__file__).resolve().parent
snapshot=json.loads((OUT/'snapshot.json').read_text())
assert file_hash(OUT/'checkpoint-current.npz')==snapshot['checkpoint_sha256']
assert scientific_source_hash(repository_root())==snapshot['source_sha256']
assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
with np.load(OUT/'checkpoint-current.npz',allow_pickle=False) as saved:
    identity=json.loads(saved['metadata'].item())['identity'];weights=saved['parameters']
rows=[json.loads(s) for s in (OUT/'test-selection.jsonl').read_text().splitlines()]
assert len(rows)==39
assert file_hash(OUT/'test-selection.jsonl')=='1713f7a36966393269fc9bb2b5375033aca6b2b46191c4ba831ca13ce9c3c425'
def fingerprint(row):
    xyz=np.asarray(row['coords_angstrom']);symbols=row['symbols']
    pairs=sorted((tuple(sorted((symbols[i],symbols[j]))),round(float(np.linalg.norm(xyz[i]-xyz[j])),5))
        for i in range(len(xyz)) for j in range(i))
    return hashlib.sha256(json.dumps([sorted(symbols),row['charge'],row['spin'],pairs]).encode()).hexdigest()
training=identity['references']['structures'];training_signatures={fingerprint(r) for r in training}
overlap=[r['structure_id'] for r in rows if fingerprint(r) in training_signatures]
assert not overlap,overlap
formula=lambda r:tuple(sorted(Counter(r['symbols']).items()))
training_formulas={formula(r) for r in training}
cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
initial,predict,_,_=make_mace_predictor(identity['seed'],identity['model_config'],return_mask=True)
assert initial.shape==weights.shape
cfg=RKSConfig(xc_spec='hf',jk_backend='df',max_cycle=200,conv_tol=1e-12,
    conv_tol_density=1e-10,conv_tol_grad=1e-9)

def solve(s,h,b,enuc,nelectron,atom_mask,occupation):
    n=s.shape[0]
    inputs=dict(overlap=s,hcore=h,eri=None,df_factors=b,nelectron=nelectron,
        nuclear_repulsion=enuc,ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
    result,rescued,cycles=_run_scf_with_rescue(run_rks_from_integrals_traceable,inputs,cfg,
        rescue_level_shift=.2,fallback_guess=lambda:atomic_density_guess(s,atom_mask,occupation))
    result,stability=stabilize_df_rhf(result,config=cfg,**inputs)
    d=result.density_matrix;j,k=build_jk_from_df(b,d);f=h+j-.5*k
    energy=jnp.sum(d*h)+.5*jnp.sum(d*j)-.25*jnp.sum(d*k)+enuc
    _,c=_diagonalize_fock(f,_orthogonalizer(s,cfg.orthogonalization_eps))
    info=dict(converged=result.converged,rescue_used=rescued,scf_cycles=result.cycles,
        total_scf_cycles=cycles+stability['stability_scf_cycles'],
        fixed_point_residual=jnp.linalg.norm(_build_density_from_occ(c,result.mo_occ)-d),
        orbital_residual=jnp.linalg.norm(f@d@s-s@d@f),min_overlap_eigenvalue=jnp.linalg.eigvalsh(s)[0],
        reconstruction_error=jnp.abs(energy-result.total_energy),**stability)
    return energy,info
solve=jax.jit(solve,static_argnums=(4,))
started=time.perf_counter()
report=dict(status='running',checkpoint=snapshot,source_sha256=snapshot['source_sha256'],
    native_sha256=file_hash(register_integrals()._name),device=gpu.device_kind,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
    method='DF-RHF',auxbasis=identity['auxbasis'],conv_tol=cfg.conv_tol,conv_tol_grad=cfg.conv_tol_grad,
    max_cycle=cfg.max_cycle,training_distance_overlap=overlap,
    training_formula_overlap_count=sum(formula(r) in training_formulas for r in rows),
    performed_gradients=False,requested_four_center_eri=False,results=[])
exports=[]
for row in rows:
    print('BEGIN',row['structure_id'],flush=True)
    atom=list(zip(row['symbols'],row['coords_angstrom']))
    with jax.default_device(cpu):
        layout=bind_pool_candidate(prepare_direct_basis(atom,basis_family=identity['references']['basis_family'],
            core_primitives=identity['references']['core_primitives'],cart=False),identity['candidate'])
        at,ap=integrals.prepare_basis(atom,identity['auxbasis'],cart=False)
        plan=integrals.make_plan(layout.topology,backend='native')
        aux=integrals.make_auxiliary_plan(layout.topology,at)
        metric=aux.metric_factor(layout.parameters,ap)
        mask,occupation=atomic_guess_data(layout)
    graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=identity['model_config']['elements'],
        cutoff=identity['model_config'].get('cutoff',5.))
    for label,vector in [('NNAO-500-initial',initial),('NNAO-500-current',weights)]:
        try:
            tick=time.perf_counter();outputs=jax.block_until_ready(predict(jnp.asarray(vector),graph))
            nn_seconds=time.perf_counter()-tick
            with jax.default_device(cpu):
                bound=layout.bind(jax.device_put(outputs,cpu));tick=time.perf_counter()
                s=plan.evaluate('overlap',bound)
                h=plan.evaluate('kinetic',bound)+plan.evaluate('nuclear',bound)
                packed=aux.factors(bound,ap,metric_factor=metric)
                b=unpack_factors(packed,layout.topology.nao)
                enuc=scf.nuclear_repulsion_energy(bound.nuclear_coords,jnp.asarray(layout.topology.nuclear_charges))
                jax.block_until_ready((s,h,b,enuc));integral_seconds=time.perf_counter()-tick
                shells=layout.atom_shells(bound)
            physical=jax.device_put((s,h,b,enuc,mask,occupation),gpu)
            tick=time.perf_counter()
            energy,info=jax.block_until_ready(solve(*physical[:4],layout.nelectron,*physical[4:]))
            details=BasisRHF._checked_info(energy,info)
            result=dict(structure_id=row['structure_id'],basis=label,status='passed',energy_hartree=float(energy),
                nao=layout.topology.nao,primitive_nao=sum((2*l+1)*n for l,n in zip(
                    layout.topology.angular_momenta,layout.topology.primitive_counts)),
                nn_seconds=nn_seconds,integral_seconds=integral_seconds,scf_seconds=time.perf_counter()-tick,scf=details)
            report['results'].append(result)
            exports.append(dict(structure_id=row['structure_id'],basis=label,symbols=row['symbols'],
                coords_angstrom=row['coords_angstrom'],atom_labels=[f'{s}{i}' for i,s in enumerate(row['symbols'])],
                atom_shells=shells,nao=layout.topology.nao,native_energy_hartree=float(energy),
                native_source_sha256=snapshot['source_sha256'],checkpoint_sha256=snapshot['checkpoint_sha256'],
                checkpoint_epoch=snapshot['epoch'] if label.endswith('current') else 0,
                checkpoint_update=snapshot['update_step'] if label.endswith('current') else 0,
                checkpoint_profile=label,model_training_structure_count=500 if label.endswith('current') else 0))
            print(label,row['structure_id'],float(energy),flush=True)
        except Exception as error:
            report['results'].append(dict(structure_id=row['structure_id'],basis=label,status='failed',
                error=str(error),traceback=traceback.format_exc()))
            print('FAILED',label,row['structure_id'],str(error),flush=True)
        report['elapsed_seconds']=time.perf_counter()-started
        atomic_json(OUT/'native-results.json',report)
        (OUT/'frozen-bases.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in exports))
report.update(status='complete',passed=sum(r['status']=='passed' for r in report['results']),
    failed=sum(r['status']!='passed' for r in report['results']))
atomic_json(OUT/'native-results.json',report)
assert file_hash(OUT/'checkpoint-current.npz')==snapshot['checkpoint_sha256']
