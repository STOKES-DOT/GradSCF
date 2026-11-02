"""Per-structure same-pool DZ contraction diagnostic; no MACE weight updates."""
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

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf import integrals
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.model.nnao.integrals.pool import active_pool_basis, pool_projection
from expand import expand_valence_dz
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, file_hash, repository_root, scientific_source_hash
from gradscf.model.nnao.training.pool import minimize_reference, FreeContractionRHF
from gradscf.model.nnao.workflows.pool_calibration import _save_result

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent/'nnao-relative200-20261014'
SZOUT = OUT.parent/'nnao-free-sz-diagnostic-20261014'
REACTION_IDS = ['G2RC/2', 'BH76/37', 'S22/3']
STARTS = ['sz_best', 'reference', 'sz_best_perturbed', 'reference_perturbed']
STEPS = 500
RATE = .01
GRADIENT_TOLERANCE = 1e-5

def normalize_active(coefficients, layout):
    coefficients = np.array(coefficients, copy=True)
    mask = np.zeros_like(coefficients, dtype=bool)
    for atom, slot, count, columns in zip(layout.shell_atoms, layout.slots,
            layout.topology.primitive_counts, layout.topology.contraction_counts):
        if slot >= 0:
            mask[atom, slot:slot+columns, :count] = True
            for column in range(slot, slot+columns):
                v = coefficients[atom, column, :count]
                coefficients[atom, column, :count] = v/np.linalg.norm(v)
    return coefficients, mask

def main():
    assert jax.config.jax_enable_x64 and jax.default_backend() == 'gpu'
    snapshot = json.loads((PARENT/'snapshot.json').read_text())
    assert snapshot['checkpoint_epoch'] == 629
    assert file_hash(PARENT/'checkpoint-best.npz') == snapshot['checkpoint_sha256']
    assert scientific_source_hash(repository_root()) == snapshot['source_sha256']
    reactions = [json.loads(s) for s in (PARENT/'reactions.jsonl').read_text().splitlines()
                 if json.loads(s)['reaction_id'] in REACTION_IDS]
    ids = {t['structure_id'] for r in reactions for t in r['terms']}
    rows = [json.loads(s) for s in (PARENT/'test-selection.jsonl').read_text().splitlines()
            if json.loads(s)['structure_id'] in ids]
    assert len(rows) == 7 and len(reactions) == 3
    native = json.loads((PARENT/'native-results.json').read_text())
    baseline = {(r['structure_id'], r['basis']): r['energy_hartree']
                for r in native['results'] if r['status']=='passed'}
    configuration = dict(checkpoint_sha256=snapshot['checkpoint_sha256'],
        script_sha256=file_hash(__file__), expansion_sha256=file_hash(OUT/'expand.py'),
        free_sz_summary_sha256=file_hash(SZOUT/'summary.json'), valence_radial_channels=2,polarization_radial_channels=1,
        scientific_source_sha256=snapshot['source_sha256'], reaction_ids=REACTION_IDS,
        starts=STARTS, max_steps=STEPS, learning_rate=RATE,
        coefficient_gradient_tolerance=GRADIENT_TOLERANCE, implicit_tolerance=1e-8,
        perturbation_std=.05, seed=20261014, basis_family=snapshot['basis_family'],
        core_primitives=snapshot['core_primitives'], fixed_exponents=True,
        references_sha256=file_hash(PARENT/'test-selection.jsonl'),
        candidate_sha256=hashlib.sha256(json.dumps(snapshot['candidate'],sort_keys=True).encode()).hexdigest())
    if (OUT/'configuration.json').exists():
        assert json.loads((OUT/'configuration.json').read_text()) == configuration
    atomic_json(OUT/'configuration.json', configuration)
    atomic_json(OUT/'reactions.json', reactions)
    atomic_json(OUT/'structures.json', rows)
    sz_results={r['structure_id']:r for r in json.loads((SZOUT/'summary.json').read_text())['results']}
    cpu, gpu = jax.devices('cpu')[0], jax.devices('gpu')[0]
    report = dict(status='running', configuration=configuration,
        device=gpu.device_kind, job_id=os.environ.get('SLURM_JOB_ID'),
        cpu_count=os.environ.get('SLURM_CPUS_PER_TASK'), results=[])
    started = time.perf_counter()
    for row in rows:
        identifier = row['structure_id']
        tag = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        record = dict(structure_id=identifier, status='running', starts=[])
        report['results'].append(record)
        atomic_json(OUT/'progress.json', report)
        try:
            atomic_json(OUT/'phase.json', dict(structure_id=identifier, phase='prepare_cache'))
            with jax.default_device(cpu):
                layout = prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                    basis_family=snapshot['basis_family'],core_primitives=snapshot['core_primitives'])
                layout = bind_pool_candidate(layout, snapshot['candidate'])
            tick = time.perf_counter()
            sz_layout=layout
            with np.load(SZOUT/f'{tag}-best.npz',allow_pickle=False) as saved:
                sz_best=np.array(saved['coefficients'])
            sz_free, manifest=prepare_fixed_pool(row,sz_layout,SZOUT/'cache'/f'{tag}.npz',
                                                storage='device',implicit_tolerance=1e-8)
            with jax.default_device(cpu):
                layout=expand_valence_dz(sz_layout,sz_best)
                reference_layout=expand_valence_dz(sz_layout,sz_layout.reference_outputs())
                best_seed,mask=normalize_active(np.asarray(layout.reference_outputs()),layout)
                reference,_=normalize_active(np.asarray(reference_layout.reference_outputs()),layout)
                assert active_pool_basis(layout)[0]==active_pool_basis(sz_layout)[0]
                sz_projection=np.asarray(pool_projection(sz_layout,sz_best))
                dz_projection=np.asarray(pool_projection(layout,best_seed))
                columns=[];offset=0
                for l,nc in zip(layout.topology.angular_momenta,layout.topology.contraction_counts):
                    columns.extend(range(offset,offset+2*l+1));offset+=nc*(2*l+1)
                np.testing.assert_allclose(dz_projection[:,columns],sz_projection,atol=1e-12,rtol=1e-12)
            free=FreeContractionRHF(layout,sz_free.raw,sz_free.enuc,sz_free.value,sz_free.gradient,
                                   identifier,sz_free.diagnostics)
            # The expanded contraction must reproduce the native AO matrix ordering.
            projected=jax.block_until_ready(free.project(jax.device_put(best_seed,gpu),*free.raw))
            with jax.default_device(cpu):
                bound=layout.bind(jax.device_put(best_seed,cpu))
                plan=integrals.make_plan(layout.topology,backend='native')
                ns=plan.evaluate('overlap',bound)
                nh=plan.evaluate('kinetic',bound)+plan.evaluate('nuclear',bound)
                at,ap=integrals.prepare_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                                             'def2-universal-jkfit',cart=False)
                nb=unpack_factors(integrals.make_auxiliary_plan(layout.topology,at).factors(bound,ap),layout.topology.nao)
                matrix_errors={}
                for key,a,b in zip(['overlap','hcore','df_factors'],projected,[ns,nh,nb]):
                    a,b=np.asarray(a),np.asarray(b)
                    matrix_errors[key]=float(np.max(np.abs(a-b)))
                    np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-10)
            record.update(cache_seconds=time.perf_counter()-tick,cache_bytes=manifest['bytes'],
                          sz_nao=sz_layout.topology.nao,nao=layout.topology.nao,
                          active_coefficient_count=int(mask.sum()),matrix_errors=matrix_errors,
                          free_sz_energy_hartree=sz_results[identifier]['best_energy_hartree'])
            calls = 0
            active_start = 'gates'
            def evaluate(c, gradient=True):
                nonlocal calls
                result = free.evaluate_device(jax.device_put(c,gpu),gradient=gradient)
                calls += 1
                if calls % 25 == 0:
                    atomic_json(OUT/'phase.json',dict(structure_id=identifier,
                        phase=active_start, evaluations=calls,
                        last_trial_energy_hartree=result[0],
                        note='Trial values are not necessarily accepted optimizer states.'))
                return result[0], None if result[1] is None else np.asarray(result[1]),result[2]
            atomic_json(OUT/'phase.json', dict(structure_id=identifier,phase='coefficient_gradient_gate'))
            tick = time.perf_counter()
            e0, g0, info0 = evaluate(best_seed)
            e_ref, _, info_ref = evaluate(reference)
            assert e0 <= record['free_sz_energy_hartree']+1e-7, 'Nested DZ SCF above free SZ: inspect SCF branch'
            assert e_ref <= baseline[(identifier,'NNAO-200-initial')]+1e-7, 'Nested reference DZ above reference SZ'
            rng = np.random.default_rng(20261014)
            direction = rng.normal(size=best_seed.shape)*mask
            direction /= np.linalg.norm(direction)
            ad = float(np.sum(g0*direction)); checks=[]
            for h in [1e-4, 3e-5]:
                plus = evaluate(best_seed+h*direction,gradient=False)[0]
                minus = evaluate(best_seed-h*direction,gradient=False)[0]
                fd = (plus-minus)/(2*h)
                checks.append(dict(step=h,implicit_directional=ad,finite_difference=fd,
                    absolute_error=abs(ad-fd),passed=abs(ad-fd)<=2e-6+2e-3*abs(ad)))
            record.update(dz_initial_energy_hartree=e0,reference_energy_hartree=e_ref,
                initial_gradient_norm=float(np.linalg.norm(g0)),initial_scf=info0,
                gradient_checks=checks,gate_seconds=time.perf_counter()-tick)
            atomic_json(OUT/'progress.json',report)
            assert all(c['passed'] for c in checks),'Coefficient finite-difference gate failed'
            candidates = [(e0,'expanded_sz_best',best_seed), (e_ref,'expanded_reference',reference)]
            for index, name in enumerate(STARTS):
                active_start = name
                atomic_json(OUT/'phase.json',dict(structure_id=identifier,phase=name,evaluations=calls))
                point = np.array(best_seed if name.startswith('sz_best') else reference,copy=True)
                if name.endswith('perturbed'):
                    point += .05*np.random.default_rng(20261014+index).normal(size=point.shape)*mask
                    point, _ = normalize_active(point,layout)
                tick = time.perf_counter()
                try:
                    result = minimize_reference(point,evaluate,max_steps=STEPS,learning_rate=RATE,
                        gradient_tolerance=GRADIENT_TOLERANCE,max_backtracks=10)
                    path=OUT/'checkpoints'/f'{tag}-{name}.npz'
                    _save_result(path,result,kind='free_dz_diagnostic',
                        identity=dict(configuration,structure_id=identifier,start=name))
                    accepted_energies=[h['energy_hartree'] for h in result['history'] if h['accepted']]
                    assert np.all(np.diff(accepted_energies)<=1.1e-12)
                    summary={k:result[k] for k in ['status','accepted_steps','initial_energy_hartree',
                        'energy_hartree','gradient_norm','optimizer_converged']}
                    summary.update(start=name,elapsed_seconds=time.perf_counter()-tick,
                                   checkpoint=str(path.relative_to(OUT)))
                    record['starts'].append(summary)
                    candidates.append((result['energy_hartree'],name,result['parameters']))
                except Exception as error:
                    record['starts'].append(dict(start=name,status='failed',error=str(error),
                                                traceback=traceback.format_exc()))
                atomic_json(OUT/'progress.json',report)
                print(identifier,name,record['starts'][-1],flush=True)
            energy, name, parameters = min(candidates,key=lambda c:c[0])
            good=[s for s in record['starts'] if s['status']!='failed']
            converged=[s for s in good if s['optimizer_converged']]
            record.update(status='complete' if len(good)==len(STARTS) else 'partial',
                best_energy_hartree=energy,best_candidate=name,
                initial_dz_to_best_millihartree=1000*(e0-energy),
                free_sz_to_dz_millihartree=1000*(record['free_sz_energy_hartree']-energy),
                completed_starts=len(good),converged_starts=len(converged),
                endpoint_spread_millihartree=1000*(max(s['energy_hartree'] for s in good)-min(s['energy_hartree'] for s in good)) if good else None,
                converged_spread_millihartree=1000*(max(s['energy_hartree'] for s in converged)-min(s['energy_hartree'] for s in converged)) if converged else None)
            np.savez(OUT/f'{tag}-best.npz',coefficients=parameters)
            # Recheck selected best using the same SCF and export a frozen basis.
            checked_energy, checked_gradient, checked_info=evaluate(parameters)
            np.testing.assert_allclose(checked_energy,energy,atol=1e-9,rtol=0)
            with jax.default_device(cpu):
                shells=layout.atom_shells(layout.bind(jax.device_put(parameters,cpu)))
            atomic_json(OUT/f'{tag}-best-basis.json',dict(structure_id=identifier,
                symbols=row['symbols'],coords_angstrom=row['coords_angstrom'],atom_shells=shells,
                energy_hartree=energy,gradient_norm=float(np.linalg.norm(checked_gradient)),scf=checked_info))
        except Exception as error:
            record.update(status='failed',error=str(error),traceback=traceback.format_exc())
            print('FAILED',identifier,str(error),flush=True)
        report['elapsed_seconds']=time.perf_counter()-started
        atomic_json(OUT/'progress.json',report)
        jax.clear_caches();gc.collect()
    report['status']='complete' if all(r['status']=='complete' for r in report['results']) else 'complete_with_failures'
    assert file_hash(PARENT/'checkpoint-best.npz') == snapshot['checkpoint_sha256']
    atomic_json(OUT/'summary.json',report)
    atomic_json(OUT/'progress.json',report)
    atomic_json(OUT/'phase.json',dict(phase='complete',status=report['status']))

if __name__ == '__main__':
    main()
