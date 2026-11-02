"""Frozen current-checkpoint benchmark; no fitting, ERI tensor or gradients."""
import gc
import json
import os
from pathlib import Path
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals,scf
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.runtime import atomic_json,file_hash,scientific_source_hash,repository_root,verify_source_imports
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.model.nnao.training.objective import _run_scf_with_rescue,BasisRHF
from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable
from gradscf.scf.stability import stabilize_df_rhf
from gradscf.scf.core import _build_density_from_occ,_diagonalize_fock,_orthogonalizer

OUT=Path(__file__).resolve().parent
snapshot=json.loads((OUT/'snapshot.json').read_text())
assert file_hash(OUT/'checkpoint-best.npz')==snapshot['checkpoint_sha256']
assert scientific_source_hash(repository_root())==snapshot['source_sha256']
assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
with np.load(OUT/'checkpoint-best.npz',allow_pickle=False) as saved:
    metadata=json.loads(str(saved['metadata'].item()))
    weights=np.array(saved['parameters'])
identity=metadata['identity'];del metadata
rows=[json.loads(s) for s in (OUT/'test-selection.jsonl').read_text().splitlines()]
assert len(rows)==10
assert file_hash(OUT/'test-selection.jsonl')=='60c3a44e19214decdbc07701c27460b8098e38dc5184e4f44e0e8e14ceba8987'
training=identity['references']['structures']
formula=lambda row:tuple(sorted(Counter(row['symbols']).items()))
assert not {r['structure_id'] for r in rows}&{r['structure_id'] for r in training}
assert not {formula(r) for r in rows}&{formula(r) for r in training}
cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
initial,predict,_,_=make_mace_predictor(identity['seed'],identity['model_config'],return_mask=True)
assert initial.shape==weights.shape
cfg=RKSConfig(xc_spec='hf',jk_backend='df',max_cycle=200,conv_tol=1e-12,
              conv_tol_density=1e-10,conv_tol_grad=1e-9)

def solve(s,h,b,enuc,nelectron):
    n=s.shape[0]
    inputs=dict(overlap=s,hcore=h,eri=None,df_factors=b,nelectron=nelectron,
        nuclear_repulsion=enuc,ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
    result,rescued,cycles=_run_scf_with_rescue(run_rks_from_integrals_traceable,inputs,cfg,rescue_level_shift=.2)
    result,stability=stabilize_df_rhf(result,config=cfg,**inputs)
    d=result.density_matrix;j,k=build_jk_from_df(b,d);f=h+j-.5*k
    energy=jnp.sum(d*h)+.5*jnp.sum(d*j)-.25*jnp.sum(d*k)+enuc
    _,c=_diagonalize_fock(f,_orthogonalizer(s,cfg.orthogonalization_eps))
    info=dict(converged=result.converged,rescue_used=rescued,scf_cycles=result.cycles,
        total_scf_cycles=cycles+stability['stability_scf_cycles'],
        fixed_point_residual=jnp.linalg.norm(_build_density_from_occ(c,result.mo_occ)-d),
        orbital_residual=jnp.linalg.norm(f@d@s-s@d@f),
        min_overlap_eigenvalue=jnp.linalg.eigvalsh(s)[0],reconstruction_error=jnp.abs(energy-result.total_energy),
        **stability)
    return energy,info

solve=jax.jit(solve,static_argnums=(4,))
report=dict(status='running',checkpoint=snapshot,structures_sha256=file_hash(OUT/'test-selection.jsonl'),
    imported_sources=verify_source_imports(repository_root()),devices=[str(d) for d in jax.devices()],
    device_kinds=[d.device_kind for d in jax.devices()],slurm_job_id=os.environ.get('SLURM_JOB_ID'),
    source_sha256=snapshot['source_sha256'],performed_gradients=False,requested_four_center_eri=False,
    training_id_overlap=0,training_formula_overlap=0,results=[])
exports=[json.loads(s) for s in (OUT/'old24-frozen-bases.jsonl').read_text().splitlines()]
assert len(exports)==10 and all(r['basis']=='NNAO-pilot-best975' for r in exports)
for row in rows:
    print('BEGIN',row['structure_id'],flush=True)
    atom=list(zip(row['symbols'],row['coords_angstrom']))
    with jax.default_device(cpu):
        layout=prepare_direct_basis(atom,basis_family=identity['references']['basis_family'],
                                    core_primitives=identity['references']['core_primitives'],cart=False)
        layout=bind_pool_candidate(layout,identity['candidate'])
    graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=identity['model_config']['elements'])
    for label,vector in [('NNAO-200-initial',initial),('NNAO-200-epoch594',weights)]:
        start=time.perf_counter();nn_times=[]
        for _ in range(2):
            tick=time.perf_counter();outputs=predict(jnp.asarray(vector),graph)
            outputs.block_until_ready();nn_times.append(time.perf_counter()-tick)
        with jax.default_device(cpu):
            bound=layout.bind(jax.device_put(outputs,cpu))
            tick=time.perf_counter()
            plan=integrals.make_plan(layout.topology,backend='native')
            s=plan.evaluate('overlap',bound)
            h=plan.evaluate('kinetic',bound)+plan.evaluate('nuclear',bound)
            at,ap=integrals.prepare_basis(atom,'def2-universal-jkfit',cart=False)
            packed=integrals.make_auxiliary_plan(layout.topology,at).factors(bound,ap)
            b=unpack_factors(packed,layout.topology.nao)
            enuc=scf.nuclear_repulsion_energy(bound.nuclear_coords,jnp.asarray(layout.topology.nuclear_charges))
            jax.block_until_ready((s,h,b,enuc));integral_seconds=time.perf_counter()-tick
            shells=layout.atom_shells(bound)
        tick=time.perf_counter();physical=jax.device_put((s,h,b,enuc),gpu)
        jax.block_until_ready(physical);transfer=time.perf_counter()-tick
        timings=[];energies=[]
        for repeat in range(2):
            tick=time.perf_counter();energy,info=solve(*physical,layout.nelectron)
            jax.block_until_ready((energy,info));elapsed=time.perf_counter()-tick
            details=BasisRHF._checked_info(energy,info);energies.append(float(energy))
            timings.append(dict(index=repeat,seconds=elapsed,scf=details))
        np.testing.assert_allclose(energies,energies[0],atol=1e-9,rtol=0)
        result=dict(structure_id=row['structure_id'],basis=label,status='passed',energy_hartree=energies[-1],
            nao=layout.topology.nao,primitive_nao=sum((2*l+1)*n for l,n in zip(layout.topology.angular_momenta,layout.topology.primitive_counts)),
            fock_shape=[layout.topology.nao]*2,nn_seconds=nn_times,scf_timings=timings,
            integral_seconds=integral_seconds,host_device_transfer_seconds=transfer,elapsed_seconds=time.perf_counter()-start)
        labels=[f'{s}{i}' for i,s in enumerate(row['symbols'])]
        frozen=dict(structure_id=row['structure_id'],basis=label,symbols=row['symbols'],coords_angstrom=row['coords_angstrom'],
            atom_labels=labels,atom_shells=shells,nao=layout.topology.nao,
            native_energy_hartree=energies[-1],native_source_sha256=snapshot['source_sha256'],
            checkpoint_sha256=snapshot['checkpoint_sha256'],checkpoint_profile='current-fixed-pool' if label.endswith('594') else 'current-pool-initial',
            checkpoint_epoch=594 if label.endswith('594') else 0,model_training_structure_count=200 if label.endswith('594') else 0)
        report['results'].append(result);exports.append(frozen)
        atomic_json(OUT/'native-results.json',report)
        print(label,row['structure_id'],energies[-1],flush=True)
        del s,h,b,packed,physical,enuc
    jax.clear_caches();gc.collect()
report['status']='complete';report['passed']=len(report['results'])
assert report['passed']==20
assert file_hash(OUT/'checkpoint-best.npz')==snapshot['checkpoint_sha256']
atomic_json(OUT/'native-results.json',report)
(OUT/'frozen-bases.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in exports))
