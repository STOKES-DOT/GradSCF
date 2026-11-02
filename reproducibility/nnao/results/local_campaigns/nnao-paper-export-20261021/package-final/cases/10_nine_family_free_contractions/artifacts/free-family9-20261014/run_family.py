"""One fixed-pool member; finite-budget free contractions, no MACE training."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import integrals
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.integrals.pool import active_pool_basis
from gradscf.model.nnao.training.pool import FreeContractionRHF,minimize_reference
from gradscf.model.nnao.workflows.pool_calibration import _save_result
from gradscf.model.nnao.runtime import atomic_json,file_hash,repository_root,scientific_source_hash
from family import lift,normalize,active_mask

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'nnao-free-sz-diagnostic-20261014'
OLD_DZ=ROOT.parent/'nnao-free-dz-diagnostic-20261014'
PARENT=ROOT.parent/'nnao-relative200-20261014'

def main(tier,zeta):
    assert jax.config.jax_enable_x64 and jax.default_backend()=='gpu'
    provenance=json.loads((ROOT/'provenance.json').read_text())
    assert scientific_source_hash(repository_root())==provenance['scientific_source_sha256']
    candidate=json.loads((ROOT/f'inputs/{tier}-candidate.json').read_text())
    rows=json.loads((OLD/'structures.json').read_text())
    name=f'{tier}-{zeta}';out=ROOT/'results'/name;out.mkdir(parents=True,exist_ok=True)
    label=f'NNAO-free-{name}'
    identity=dict(tier=tier,zeta=zeta,family=f'nnao_{("szp","dzp","tzp")[zeta-1]}_{tier}',
        core_primitives=6,source_sha256=provenance['scientific_source_sha256'],
        script_sha256=file_hash(__file__),family_helper_sha256=file_hash(ROOT/'family.py'),
        pool_sha256=file_hash(ROOT/f'inputs/{tier}-candidate.json'),
        structures_sha256=file_hash(OLD/'structures.json'),steps=500,rate=.01,
        starts=['warm','reference','warm_perturbed','reference_perturbed'],
        gradient_tolerance=1e-5,implicit_tolerance=1e-8,
        full_span_valence_policy='Freeze square invertible columns; only change of basis, no missing variational freedom.')
    if (out/'configuration.json').exists():
        assert json.loads((out/'configuration.json').read_text())==identity,'Configuration/source changed'
    atomic_json(out/'configuration.json',identity)
    if (out/'summary.json').exists():
        assert json.loads((out/'summary.json').read_text())['status']=='complete','Inspect prior incomplete member'
        print('ALREADY_COMPLETE',name,flush=True);return
    old_sz={r['structure_id']:r for r in json.loads((OLD/'summary.json').read_text())['results']}
    old_dz={r['structure_id']:r for r in json.loads((OLD_DZ/'summary.json').read_text())['results']}
    report=dict(status='running',identity=identity,device=jax.devices('gpu')[0].device_kind,
        job_id=os.environ.get('SLURM_JOB_ID'),cpu_count=os.environ.get('SLURM_CPUS_PER_TASK'),results=[])
    cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
    started=time.perf_counter();exports=[]
    for row in rows:
        sid=row['structure_id'];tag=hashlib.sha256(sid.encode()).hexdigest()[:16]
        r=dict(structure_id=sid,status='running',starts=[]);report['results'].append(r)
        atomic_json(out/'progress.json',report)
        try:
            atomic_json(ROOT/'phase.json',dict(member=name,structure_id=sid,phase='prepare_fixed_pool'))
            with jax.default_device(cpu):
                sz=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                    basis_family=f'nnao_szp_{tier}',core_primitives=6)
                sz=bind_pool_candidate(sz,candidate)
                layout=lift(sz,sz.reference_outputs(),zeta)
                reference=normalize(np.asarray(layout.reference_outputs()),layout)
                warm=np.array(reference,copy=True);lower_energy=None
                if zeta>1:
                    previous=ROOT/'results'/f'{tier}-{zeta-1}'
                    lower_record=next(x for x in json.loads((previous/'summary.json').read_text())['results'] if x['structure_id']==sid)
                    assert lower_record['status']=='complete'
                    with np.load(previous/f'{tag}-best.npz',allow_pickle=False) as f:c=np.array(f['coefficients'])
                    lower=lift(sz,sz.reference_outputs(),zeta-1)
                    warm=np.asarray(lift(lower,c,zeta).reference_outputs())
                    lower_energy=lower_record['best_energy_hartree']
                elif tier=='l':
                    with np.load(OLD/f'{tag}-best.npz',allow_pickle=False) as f:warm=np.array(f['coefficients'])
                else:
                    warm=reference+.025*np.random.default_rng(20261014).normal(size=reference.shape)*active_mask(layout)
                warm=normalize(warm,layout)
            cache=ROOT/'cache'/tier;cache.mkdir(parents=True,exist_ok=True)
            tick=time.perf_counter()
            base,manifest=prepare_fixed_pool(row,sz,cache/f'{tag}.npz',storage='device',implicit_tolerance=1e-8)
            assert active_pool_basis(layout)[0]==active_pool_basis(sz)[0]
            free=FreeContractionRHF(layout,base.raw,base.enuc,base.value,base.gradient,sid,base.diagnostics)
            mask=active_mask(layout);full_mask=np.zeros_like(mask)
            for atom,slot,role,n,c in zip(layout.shell_atoms,layout.slots,layout.roles,
                    layout.topology.primitive_counts,layout.topology.contraction_counts):
                if slot>=0 and role.startswith('valence_') and n==c:full_mask[atom,slot:slot+c,:n]=True
            r.update(nao=layout.topology.nao,pool_nao=manifest['pool_nao'],cache_bytes=manifest['bytes'],
                cache_seconds=time.perf_counter()-tick,optimized_coefficients=int(mask.sum()),
                full_span_fixed_coefficients=int(full_mask.sum()))
            projected=jax.block_until_ready(free.project(jax.device_put(warm,gpu),*free.raw))
            with jax.default_device(cpu):
                bound=layout.bind(jax.device_put(warm,cpu));plan=integrals.make_plan(layout.topology,backend='native')
                s=plan.evaluate('overlap',bound);h=plan.evaluate('kinetic',bound)+plan.evaluate('nuclear',bound)
                at,ap=integrals.prepare_basis(list(zip(row['symbols'],row['coords_angstrom'])),'def2-universal-jkfit',cart=False)
                b=unpack_factors(integrals.make_auxiliary_plan(layout.topology,at).factors(bound,ap),layout.topology.nao)
                errors={}
                for key,a,v in zip(['overlap','hcore','df_factors'],projected,[s,h,b]):
                    a,v=np.asarray(a),np.asarray(v);errors[key]=float(np.max(np.abs(a-v)))
                    np.testing.assert_allclose(a,v,atol=1e-8,rtol=1e-10)
                del s,h,b
            r['matrix_errors']=errors
            calls=0;active_start='gates'
            def evaluate(c,gradient=True):
                nonlocal calls
                e,g,info=free.evaluate_device(jax.device_put(c,gpu),gradient=gradient)
                calls+=1
                if calls%50==0:
                    atomic_json(ROOT/'phase.json',dict(member=name,structure_id=sid,phase=active_start,
                        evaluations=calls,last_trial_energy_hartree=e))
                return e,None if g is None else np.asarray(g)*mask,info
            tick=time.perf_counter();e0,g0,info0=evaluate(warm)
            eref,_,_=evaluate(reference)
            r.update(initial_energy_hartree=e0,reference_energy_hartree=eref,initial_scf=info0)
            if lower_energy is not None:
                assert e0<=lower_energy+1e-7,'Expanded SCF above lower-zeta best; inspect branch'
                r['lower_zeta_energy_hartree']=lower_energy
            if tier=='l' and zeta==1:
                np.testing.assert_allclose(e0,old_sz[sid]['best_energy_hartree'],atol=1e-7,rtol=0)
                r['historical_sz_replay_error']=abs(e0-old_sz[sid]['best_energy_hartree'])
            extra=[]
            if tier=='l' and zeta==2:
                with np.load(OLD_DZ/f'{tag}-best.npz',allow_pickle=False) as f:old=np.array(f['coefficients'])
                old_energy,_,_=evaluate(old)
                np.testing.assert_allclose(old_energy,old_dz[sid]['best_energy_hartree'],atol=1e-7,rtol=0)
                extra=[(old_energy,'historical_DZ',old)]
                r['historical_dz_replay_error']=abs(old_energy-old_dz[sid]['best_energy_hartree'])
            if full_mask.any():
                _,raw_gradient,_=free.evaluate_device(jax.device_put(warm,gpu))
                nullnorm=float(np.linalg.norm(np.asarray(raw_gradient)[full_mask]))
                assert nullnorm<2e-6,'Nonzero gradient along complete valence span'
                probe=np.array(warm,copy=True);rng=np.random.default_rng(81)
                for atom,slot,role,n,c in zip(layout.shell_atoms,layout.slots,layout.roles,
                        layout.topology.primitive_counts,layout.topology.contraction_counts):
                    if slot>=0 and role.startswith('valence_') and n==c:
                        transform=np.eye(c)+.05*rng.normal(size=(c,c))
                        assert abs(np.linalg.det(transform))>.1
                        probe[atom,slot:slot+c,:n]=(probe[atom,slot:slot+c,:n].T@transform).T
                ep=evaluate(probe,gradient=False)[0]
                np.testing.assert_allclose(ep,e0,atol=1e-8,rtol=0)
                r.update(full_span_gradient_norm=nullnorm,full_span_energy_invariance_error=abs(ep-e0))
            direction=np.random.default_rng(20261014).normal(size=warm.shape)*mask
            direction/=np.linalg.norm(direction);ad=float(np.sum(g0*direction));checks=[]
            for step in [1e-4,3e-5]:
                fd=(evaluate(warm+step*direction,False)[0]-evaluate(warm-step*direction,False)[0])/(2*step)
                checks.append(dict(step=step,ad=ad,fd=fd,error=abs(ad-fd),passed=abs(ad-fd)<=2e-6+2e-3*abs(ad)))
            r.update(gradient_checks=checks,gate_seconds=time.perf_counter()-tick)
            assert all(c['passed'] for c in checks),'Directional derivative gate failed'
            candidates=[(e0,'warm_initial',warm),(eref,'reference_initial',reference),*extra]
            for index,name_start in enumerate(identity['starts']):
                active_start=name_start
                point=np.array(warm if name_start.startswith('warm') else reference,copy=True)
                if name_start.endswith('perturbed'):
                    point=normalize(point+.05*np.random.default_rng(20261014+index).normal(size=point.shape)*mask,layout)
                tick=time.perf_counter()
                atomic_json(ROOT/'phase.json',dict(member=name,structure_id=sid,phase=active_start))
                try:
                    optimized=minimize_reference(point,evaluate,max_steps=500,learning_rate=.01,
                        gradient_tolerance=1e-5,max_backtracks=10)
                    path=out/'checkpoints'/f'{tag}-{name_start}.npz'
                    _save_result(path,optimized,kind='free_family9',identity=dict(identity,structure_id=sid,start=name_start))
                    ep=[x['energy_hartree'] for x in optimized['history'] if x['accepted']]
                    assert np.all(np.diff(ep)<=1.1e-12)
                    state={k:optimized[k] for k in ['status','energy_hartree','gradient_norm','accepted_steps','optimizer_converged']}
                    state.update(start=name_start,seconds=time.perf_counter()-tick,checkpoint=str(path.relative_to(out)))
                    candidates.append((optimized['energy_hartree'],name_start,optimized['parameters']))
                except Exception as error:
                    state=dict(start=name_start,status='failed',error=str(error),traceback=traceback.format_exc())
                r['starts'].append(state);atomic_json(out/'progress.json',report)
                print(name,sid,state,flush=True)
            energy,best,parameters=min(candidates,key=lambda x:x[0])
            checked,grad,info=evaluate(parameters)
            np.testing.assert_allclose(checked,energy,atol=1e-9,rtol=0)
            good=[x for x in r['starts'] if x['status']!='failed']
            r.update(status='complete' if len(good)==4 else 'partial',best_energy_hartree=energy,best_candidate=best,
                best_gradient_norm=float(np.linalg.norm(grad)),converged_starts=sum(x['optimizer_converged'] for x in good),
                endpoint_spread_millihartree=1000*(max(x['energy_hartree'] for x in good)-min(x['energy_hartree'] for x in good)) if good else None)
            np.savez(out/f'{tag}-best.npz',coefficients=parameters)
            with jax.default_device(cpu):shells=layout.atom_shells(layout.bind(jax.device_put(parameters,cpu)))
            exports.append(dict(structure_id=sid,basis=label,symbols=row['symbols'],coords_angstrom=row['coords_angstrom'],
                atom_labels=[f'{s}{i}' for i,s in enumerate(row['symbols'])],atom_shells=shells,nao=layout.topology.nao,
                native_energy_hartree=energy,native_source_sha256=identity['source_sha256'],
                checkpoint_profile=f'free-contraction-{name}',pool_sha256=identity['pool_sha256'],model_training_structure_count=0))
            (out/'frozen-bases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in exports))
        except Exception as error:
            r.update(status='failed',error=str(error),traceback=traceback.format_exc());print('FAILED',name,sid,str(error),flush=True)
        report['elapsed_seconds']=time.perf_counter()-started
        atomic_json(out/'progress.json',report);jax.clear_caches();gc.collect()
    (out/'frozen-bases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in exports))
    report['status']='complete' if all(x['status']=='complete' for x in report['results']) else 'complete_with_failures'
    atomic_json(out/'summary.json',report);atomic_json(out/'progress.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tier',choices=['s','m','l'],required=True);p.add_argument('--zeta',type=int,choices=[1,2,3],required=True)
    a=p.parse_args();main(a.tier,a.zeta)
