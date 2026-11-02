"""Free contractions with a second C valence-p radial channel, same fixed pool."""
from pathlib import Path
import gc
import json
import os
import time
import traceback
import jax
import numpy as np
from scipy.optimize import minimize

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.integrals.pool import active_pool_basis,pool_projection
from gradscf.model.nnao.training.initial_guess import atomic_guess_data
from gradscf.model.nnao.training.pool import FreeContractionRHF
from gradscf.model.nnao.runtime import atomic_json,file_hash,repository_root,scientific_source_hash
from expansion import expand_carbon_p

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'nnao-ethane-torsion-20261018'
CHECKPOINT=OUT.parent/'nnao-relative500-20261018'


def solve(row,identity,snapshot,baseline):
    started=time.perf_counter()
    folder=OUT/f"angle-{int(row['angle_deg']):03d}"
    folder.mkdir(exist_ok=True)
    atomic_json(folder/'structure.json',row)
    report=dict(status='running',angle_deg=row['angle_deg'],structure_id=row['structure_id'],
        old_energy_hartree=baseline['free_energy_hartree'],starts=[],evaluations=0)

    def progress(phase,**kwargs):
        report.update(phase=phase,elapsed_seconds=time.perf_counter()-started,**kwargs)
        atomic_json(folder/'progress.json',report)

    cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
    progress('prepare_same_pool')
    with jax.default_device(cpu):
        lower=bind_pool_candidate(prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
            basis_family=identity['references']['basis_family'],
            core_primitives=identity['references']['core_primitives'],cart=False),identity['candidate'])
        with np.load(BASE/folder.name/'best-coefficients.npz',allow_pickle=False) as saved:
            old_coefficients=np.array(saved['coefficients'])
        expanded=expand_carbon_p(lower,old_coefficients)
        reference=expand_carbon_p(lower,lower.reference_outputs())
        old_top,old_parameters=active_pool_basis(lower)
        new_top,new_parameters=active_pool_basis(expanded)
        assert old_top==new_top and expanded.topology.nao==50 and lower.topology.nao==44
        for before,after in zip(old_parameters.exponents,new_parameters.exponents):
            np.testing.assert_array_equal(before,after)
        retained=[];offset=0
        for ell,old,new in zip(lower.topology.angular_momenta,lower.topology.contraction_counts,
                              expanded.topology.contraction_counts):
            retained.extend(range(offset,offset+old*(2*ell+1)));offset+=new*(2*ell+1)
        old_t=np.asarray(pool_projection(lower,old_coefficients))
        new_t=np.asarray(pool_projection(expanded,expanded.reference_outputs()))
        nesting_error=float(np.max(np.abs(new_t[:,retained]-old_t)))
        assert nesting_error<2e-12
    tick=time.perf_counter()
    lower_problem,manifest=prepare_fixed_pool(row,lower,BASE/folder.name/'cache'/'pool.npz',
        storage='device',auxbasis=identity['auxbasis'],implicit_tolerance=1e-8)
    free=FreeContractionRHF(expanded,lower_problem.raw,lower_problem.enuc,
        lower_problem.value,lower_problem.gradient,row['structure_id'],lower_problem.diagnostics,
        atomic_guess=atomic_guess_data(expanded))
    report.update(cache_prepare_seconds=time.perf_counter()-tick,primitive_cache_reused=True,
        cache_signature=manifest['signature'],cache_sha256=manifest['cache_sha256'],
        pool_nao=manifest['pool_nao'],nao=50,nesting_transform_max_error=nesting_error,
        native_sha256=manifest['identity']['native_sha256'])
    initial=np.asarray(expanded.reference_outputs())
    other=np.asarray(reference.reference_outputs())
    mask=np.zeros_like(initial,dtype=bool)
    for atom,slot,count,columns in zip(expanded.shell_atoms,expanded.slots,
            expanded.topology.primitive_counts,expanded.topology.contraction_counts):
        if slot>=0:mask[atom,slot:slot+columns,:count]=True
    assert mask.sum()==90
    report['active_raw_coefficients']=int(mask.sum())
    timings=[]

    def evaluate(c):
        tick=time.perf_counter()
        energy,gradient,details=free.evaluate_device(jax.device_put(c,gpu),gradient=True)
        gradient=np.asarray(gradient)
        assert np.max(np.abs(gradient[~mask]),initial=0.)<1e-10
        report['evaluations']+=1;timings.append(time.perf_counter()-tick)
        return energy,gradient,details

    best=None
    for name,seed in [('lifted_44ao_best',initial),('lifted_reference',other)]:
        progress(name,iteration=0)
        tick=time.perf_counter();calls=report['evaluations'];history=[]
        accepted=None
        try:
            energy,gradient,details=evaluate(seed)
            if name=='lifted_44ao_best':
                assert energy<=baseline['free_energy_hartree']+1e-8,('Nonvariational nested SCF',energy,baseline)
                report['lifted_initial_energy_hartree']=energy
            template=seed.copy()
            current=dict(x=seed[mask].copy(),energy=energy,gradient=gradient[mask].copy(),info=details)
            accepted=[energy,seed.copy()]
            history.append(dict(iteration=0,energy_hartree=energy,gradient_norm=float(np.linalg.norm(gradient))))

            def unpack(x):
                c=template.copy();c[mask]=x;return c

            def fun(x):
                if not np.array_equal(x,current['x']):
                    value,derivative,info=evaluate(unpack(x))
                    current.update(x=np.array(x,copy=True),energy=value,gradient=derivative[mask],info=info)
                return current['energy'],current['gradient']

            def callback(x):
                energy,gradient=fun(x)
                history.append(dict(iteration=len(history),energy_hartree=energy,
                                    gradient_norm=float(np.linalg.norm(gradient))))
                if energy<accepted[0]:accepted[:]=[energy,unpack(x)]
                if len(history)%20==0:
                    np.savez(folder/f'{name}-accepted.npz',coefficients=unpack(x),gradient=gradient)
                    atomic_json(folder/f'{name}-history.json',history)
                    progress(name,iteration=len(history)-1,energy_hartree=energy,
                             gradient_norm=float(np.linalg.norm(gradient)))
                    print(row['angle_deg'],name,len(history)-1,energy,float(np.linalg.norm(gradient)),flush=True)

            result=minimize(fun,seed[mask],method='L-BFGS-B',jac=True,callback=callback,
                options=dict(maxiter=1500,maxfun=4500,maxls=30,maxcor=20,ftol=1e-15,gtol=1e-7))
            energy,gradient=fun(result.x)
            if energy<accepted[0]:accepted[:]=[energy,unpack(result.x)]
            entry=dict(start=name,status='complete',energy_hartree=energy,
                gradient_norm=float(np.linalg.norm(gradient)),gradient_converged=np.linalg.norm(gradient)<=1e-5,
                optimizer_success=bool(result.success),optimizer_message=str(result.message),
                iterations=int(result.nit),function_calls=int(result.nfev))
            entry['gradient_converged']=bool(entry['gradient_converged'])
            np.savez(folder/f'{name}-final.npz',coefficients=unpack(result.x),gradient=gradient)
        except Exception as error:
            entry=dict(start=name,status='failed',error=str(error),traceback=traceback.format_exc())
        if accepted is not None:
            np.savez(folder/f'{name}-best.npz',coefficients=accepted[1])
            if best is None or accepted[0]<best[0]:best=[accepted[0],name,accepted[1]]
        entry.update(seconds=time.perf_counter()-tick,evaluations=report['evaluations']-calls)
        report['starts'].append(entry)
        atomic_json(folder/f'{name}-history.json',history)
        progress(name+'_finished')
        print('END START',row['angle_deg'],entry,flush=True)
    if best is None:raise RuntimeError('Both expanded-basis starts failed before an accepted state.')
    energy,name,coefficients=best
    checked,gradient,info=evaluate(coefficients)
    np.testing.assert_allclose(checked,energy,atol=1e-9,rtol=0)
    assert checked<=baseline['free_energy_hartree']+1e-8
    np.savez(folder/'best-coefficients.npz',coefficients=coefficients,gradient=gradient)
    with jax.default_device(cpu):
        bound=expanded.bind(jax.device_put(coefficients,cpu))
        shells=expanded.atom_shells(bound)
    atomic_json(folder/'frozen-basis.json',dict(structure_id=row['structure_id'],
        basis='NNAO / second C p / free',symbols=row['symbols'],coords_angstrom=row['coords_angstrom'],
        atom_labels=[f'{s}{i}' for i,s in enumerate(row['symbols'])],atom_shells=shells,nao=50,
        native_energy_hartree=checked))
    good=[s for s in report['starts'] if s['status']=='complete']
    report.update(status='complete' if len(good)==2 else 'complete_with_failures',
        best_energy_hartree=checked,best_start=name,best_gradient_norm=float(np.linalg.norm(gradient)),
        best_gradient_converged=float(np.linalg.norm(gradient))<=1e-5,best_scf=info,
        energy_lowering_hartree=baseline['free_energy_hartree']-checked,
        energy_lowering_kcal_mol=627.5094740631*(baseline['free_energy_hartree']-checked),
        converged_starts=sum(s['gradient_converged'] for s in good),
        endpoint_spread_hartree=max(s['energy_hartree'] for s in good)-min(s['energy_hartree'] for s in good) if good else None,
        cold_forward_backward_seconds=timings[0],median_hot_forward_backward_seconds=float(np.median(timings[1:])))
    progress('complete')
    atomic_json(folder/'summary.json',report)
    return report


def main():
    assert jax.config.jax_enable_x64 and jax.default_backend()=='gpu'
    snapshot=json.loads((CHECKPOINT/'snapshot.json').read_text())
    assert scientific_source_hash(repository_root())==snapshot['source_sha256']
    assert file_hash(CHECKPOINT/'checkpoint-current.npz')==snapshot['checkpoint_sha256']
    with np.load(CHECKPOINT/'checkpoint-current.npz',allow_pickle=False) as saved:
        identity=json.loads(saved['metadata'].item())['identity']
    rows=json.loads((BASE/'structures.json').read_text())
    rows.sort(key=lambda row:(row['angle_deg'] not in (0.,60.),row['angle_deg']))
    baseline={r['angle_deg']:r for r in json.loads((BASE/'scan-summary.json').read_text())['points']}
    configuration=dict(script_sha256=file_hash(__file__),expansion_sha256=file_hash(OUT/'expansion.py'),
        checkpoint=snapshot,geometry_sha256=file_hash(BASE/'structures.json'),
        carbon_shells='[2s2p1d]',hydrogen_shells='[1s1p]',old_nao=44,new_nao=50,
        fixed_exponents=True,fixed_core=True,primitive_pool_unchanged=True,primitive_cache_reused=True,
        method='DF-RHF',auxbasis=identity['auxbasis'],dtype='float64',
        implicit_tolerance=1e-8,conv_tol_energy=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9,
        optimizer='L-BFGS-B; analytic implicit SCF gradients',maxiter=1500,maxfun=4500,
        maxls=30,maxcor=20,ftol=1e-15,gtol=1e-7,gradient_norm_tolerance=1e-5,
        starts=['lifted_44ao_best','lifted_reference'],finite_differences=False,
        learned_mace_head_changed=False)
    atomic_json(OUT/'configuration.json',configuration)
    started=time.perf_counter()
    scan=dict(status='running',job_id=os.environ.get('SLURM_JOB_ID'),device=jax.devices('gpu')[0].device_kind,
        cpus=os.environ.get('SLURM_CPUS_PER_TASK'),configuration=configuration,points=[])
    for row in rows:
        scan.update(phase=row['angle_deg'],elapsed_seconds=time.perf_counter()-started)
        atomic_json(OUT/'progress.json',scan)
        report=solve(row,identity,snapshot,baseline[row['angle_deg']])
        scan['points'].append({key:report[key] for key in ['angle_deg','structure_id','nao',
            'old_energy_hartree','best_energy_hartree','best_gradient_norm','best_gradient_converged',
            'energy_lowering_hartree','energy_lowering_kcal_mol','converged_starts',
            'endpoint_spread_hartree','status','elapsed_seconds']})
        scan['elapsed_seconds']=time.perf_counter()-started
        atomic_json(OUT/'progress.json',scan)
        print('END ANGLE',json.dumps(scan['points'][-1]),flush=True)
        assert report['status']=='complete' and report['best_gradient_converged'],report['starts']
        jax.clear_caches();gc.collect()
    scan['points'].sort(key=lambda r:r['angle_deg'])
    scan.update(status='complete',phase='complete',elapsed_seconds=time.perf_counter()-started)
    atomic_json(OUT/'summary.json',scan)
    atomic_json(OUT/'progress.json',scan)
    assert file_hash(CHECKPOINT/'checkpoint-current.npz')==snapshot['checkpoint_sha256']


if __name__=='__main__':main()
