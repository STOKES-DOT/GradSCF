"""Per-structure fixed-pool SZ contraction diagnostic; no MACE weight updates."""
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
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, file_hash, repository_root, scientific_source_hash
from gradscf.model.nnao.training.pool import minimize_reference
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.model.nnao.workflows.pool_calibration import _save_result

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent/'nnao-relative200-20261014'
REACTION_IDS = ['G2RC/2', 'BH76/37', 'S22/3']
STARTS = ['mace', 'reference', 'mace_perturbed', 'reference_perturbed']
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
        script_sha256=file_hash(__file__),
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
    with np.load(PARENT/'checkpoint-best.npz', allow_pickle=False) as saved:
        weights = np.array(saved['parameters'])
    _, predict, _, _ = make_mace_predictor(snapshot['seed'], snapshot['model_config'], return_mask=True)
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
            (OUT/'cache').mkdir(exist_ok=True)
            free, manifest = prepare_fixed_pool(row, layout, OUT/'cache'/f'{tag}.npz',
                                               storage='device', implicit_tolerance=1e-8)
            graph = build_graph(row['atomic_numbers'],row['coords_angstrom'],
                                element_order=snapshot['model_config']['elements'])
            mace, mask = normalize_active(np.asarray(predict(jnp.asarray(weights),graph)),layout)
            reference, _ = normalize_active(np.asarray(layout.reference_outputs()),layout)
            record.update(cache_seconds=time.perf_counter()-tick, cache_bytes=manifest['bytes'],
                          nao=layout.topology.nao, active_coefficient_count=int(mask.sum()))
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
            e0, g0, info0 = evaluate(mace)
            e_ref, _, info_ref = evaluate(reference)
            np.testing.assert_allclose(e0,baseline[(identifier,'NNAO-200-best')],atol=1e-7,rtol=0)
            np.testing.assert_allclose(e_ref,baseline[(identifier,'NNAO-200-initial')],atol=1e-7,rtol=0)
            rng = np.random.default_rng(20261014)
            direction = rng.normal(size=mace.shape)*mask
            direction /= np.linalg.norm(direction)
            ad = float(np.sum(g0*direction)); checks=[]
            for h in [1e-4, 3e-5]:
                plus = evaluate(mace+h*direction,gradient=False)[0]
                minus = evaluate(mace-h*direction,gradient=False)[0]
                fd = (plus-minus)/(2*h)
                checks.append(dict(step=h,implicit_directional=ad,finite_difference=fd,
                    absolute_error=abs(ad-fd),passed=abs(ad-fd)<=2e-6+2e-3*abs(ad)))
            record.update(mace_energy_hartree=e0,reference_energy_hartree=e_ref,
                mace_gradient_norm=float(np.linalg.norm(g0)),mace_scf=info0,
                projection_error_hartree=abs(e0-baseline[(identifier,'NNAO-200-best')]),
                gradient_checks=checks,gate_seconds=time.perf_counter()-tick)
            atomic_json(OUT/'progress.json',report)
            assert all(c['passed'] for c in checks),'Coefficient finite-difference gate failed'
            candidates = [(e0,'original_mace',mace), (e_ref,'original_reference',reference)]
            for index, name in enumerate(STARTS):
                active_start = name
                atomic_json(OUT/'phase.json',dict(structure_id=identifier,phase=name,evaluations=calls))
                point = np.array(mace if name.startswith('mace') else reference,copy=True)
                if name.endswith('perturbed'):
                    point += .05*np.random.default_rng(20261014+index).normal(size=point.shape)*mask
                    point, _ = normalize_active(point,layout)
                tick = time.perf_counter()
                try:
                    result = minimize_reference(point,evaluate,max_steps=STEPS,learning_rate=RATE,
                        gradient_tolerance=GRADIENT_TOLERANCE,max_backtracks=10)
                    path=OUT/'checkpoints'/f'{tag}-{name}.npz'
                    _save_result(path,result,kind='free_sz_diagnostic',
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
                mace_to_best_millihartree=1000*(e0-energy),
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
