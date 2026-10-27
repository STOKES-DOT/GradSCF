"""Re-evaluate selected frozen bases; preserve the original benchmark report."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnao_core.native import _solver, counterpoise, make_inputs
from nnao_core.run import FrozenMACE, spin_squared
from nnao_reactivity.run import conventional_shells
from nnao_reactivity.rescue import solve_native
from nnao_reactivity.protocols import HARTREE_TO_KCAL_MOL
from gradscf.model.nnao.runtime import atomic_json, file_hash, scientific_source_hash, repository_root

PRIORITY = ['fb81f9654e02577e0508', 'f8ce7859d72f242c10e1', 'ea90966f585e090d1442',
            '20c7cb8f37f742ca8e54', 'a51aa9ee02b4857e5efb', '791331ccd7a3b1428add']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--keys', nargs='+')
    parser.add_argument('--keys-file', type=Path)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('Enable float64 before retrying benchmark cases.')
    parent = json.loads((args.parent/'results.json').read_text())
    data = json.loads(args.dataset.read_text())
    selection = json.loads(args.selection.read_text())
    if file_hash(args.parent/'results.json') != selection['parent_sha256']:
        raise ValueError('Original numerical report has changed.')
    if file_hash(args.checkpoint) != data['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the original benchmark.')
    if args.keys and args.keys_file:
        raise ValueError('Specify keys or a keys file, not both.')
    keys = json.loads(args.keys_file.read_text()) if args.keys_file else args.keys or selection['retry_case_keys']
    if not set(keys) <= set(selection['retry_case_keys']):
        raise ValueError('Unreviewed retry case requested.')
    records = {r['key']: r for r in parent['results']}
    ordered = [k for k in PRIORITY if k in keys]+[k for k in keys if k not in PRIORITY]
    rows = [records[k] for k in ordered]
    snapshot = json.loads(args.snapshot.read_text())
    source_hash = scientific_source_hash(repository_root())
    if source_hash != snapshot['source_sha256']:
        raise ValueError('Frozen scientific source differs from the original benchmark.')
    curves = {r['identifier']: r for r in data['curves']}
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name('rescue.py'), Path(__file__).with_name('run.py'),
               Path(__file__).parents[1]/'nnao_core/native.py']
    config = dict(parent_sha256=selection['parent_sha256'], dataset_sha256=file_hash(args.dataset),
        selection_sha256=file_hash(args.selection), checkpoint_sha256=file_hash(args.checkpoint),
        keys=ordered, method='native DF-HF', auxbasis=data['auxbasis'], dtype='float64',
        scientific_source_sha256=source_hash,
        source_sha256={str(p): file_hash(p) for p in sources},
        basis_sha256={k: file_hash(args.parent/f'{k}-basis.json') for k in ordered},
        finite_differences=False, full_eri_storage=False, pyscf_solver_fallback=False,
        scf_strategy='native MINAO projection, bounded raw-polished rescue and real internal DF stability',
        stability_tolerance=1e-5, eigensolver_tolerance=1e-7, max_stability_restarts=3)
    manifest = args.output/'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Retry configuration changed; use a new output directory.')
    atomic_json(manifest, config)
    path = args.output/'results.json'
    report = json.loads(path.read_text()) if path.exists() else dict(status='running',
        job_id=os.environ.get('SLURM_JOB_ID'), device=jax.devices()[0].device_kind,
        hostname=socket.gethostname(), requested_cases=len(rows), results=[], isolated_results={})
    needs_prediction = any(r['task'] == 'curve' and r['basis'] == 'NNAO/MACE' for r in rows)
    predictor = FrozenMACE(args.checkpoint, args.snapshot) if needs_prediction else None
    old_isolated = {(r['structure_id'], r['basis']): r for r in parent['isolated_results'].values()}
    saved = {r['key'] for r in report['results']}
    started = time.perf_counter()

    def solve(data_inputs, row, shells, *, previous_energy=None):
        common = (data_inputs['overlap'], data_inputs['hcore'], data_inputs['factors'],
                  row['nelectron'], row['spin'], data_inputs['enuc'])
        result = solve_native(*common, row=row, atom_shells=shells)
        if previous_energy is not None and float(result['energy']) > previous_energy+5e-8:
            raw = _solver(row['nelectron'], row['spin'], 0.)(
                *map(jnp.asarray, common[:3]), jnp.asarray(common[-1]), None)
            result = solve_native(*common, row=row, atom_shells=shells, initial_density=raw['density_spin'])
            if float(result['energy']) > previous_energy+5e-8:
                raise RuntimeError('Native retry is higher than the retained converged parent energy.')
        return result

    def summary(state, row, overlap):
        return dict(energy_hartree=float(state['energy']), nao=len(overlap),
            orbital_residual=state['orbital_residual'], electron_numbers=state['electron_numbers'],
            spin_squared=spin_squared(state, overlap, row['nelectron'], row['spin']),
            stability=state['stability'], stability_restarts=state['stability_restarts'],
            attempts=state['attempts'], total_scf_cycles=state['total_scf_cycles'])

    def isolated(row, basis):
        key = hashlib.sha256((basis+json.dumps(row, sort_keys=True)).encode()).hexdigest()
        if key not in report['isolated_results']:
            shells = predictor.shells(row) if basis == 'NNAO/MACE' else conventional_shells(row, basis, data['basis_assets'])
            inputs = make_inputs(row, shells, auxbasis=data['auxbasis'])
            previous = old_isolated.get((row['structure_id'], basis), {}).get('energy_hartree')
            state = solve(inputs, row, shells, previous_energy=previous)
            report['isolated_results'][key] = dict(summary(state, row, inputs['overlap']),
                structure_id=row['structure_id'], basis=basis)
            atomic_json(args.output/f'{key}-isolated-basis.json', dict(row=row, atom_shells=shells))
            atomic_json(path, report)
        return report['isolated_results'][key]

    for old in rows:
        if old['key'] in saved:
            continue
        record = {k: old[k] for k in ['key', 'task', 'identifier', 'basis']}
        record.update(previous_status=old['status'], previous_energy_hartree=old.get('energy_hartree'))
        atomic_json(args.output/'progress.json', dict(**record, completed=len(report['results']), total=len(rows)))
        tick = time.perf_counter()
        print('BEGIN', json.dumps(record), flush=True)
        try:
            exported = json.loads((args.parent/f'{old["key"]}-basis.json').read_text())
            row, shells = exported['row'], exported['atom_shells']
            if old['task'] == 'species':
                inputs = make_inputs(row, shells, auxbasis=data['auxbasis'])
                state = solve(inputs, row, shells, previous_energy=old.get('energy_hartree'))
                record.update(summary(state, row, inputs['overlap']), charge=row['charge'],
                              spin=row['spin'], natoms=row['natoms'])
                np.savez_compressed(args.output/f'{old["key"]}-state.npz',
                    density_spin=state['density_spin'], mo_coeff=state['mo'], mo_occ=state['occupations'])
            else:
                definition = curves[old['identifier']]
                stages = []
                def solve_and_save_dimer(*inputs, **controls):
                    state = solve_native(*inputs, **controls)
                    if not stages:
                        np.savez_compressed(args.output/f'{old["key"]}-dimer-state.npz',
                            density_spin=state['density_spin'], mo_coeff=state['mo'], mo_occ=state['occupations'])
                    stages.append(float(state['energy']))
                    return state
                cp = counterpoise(row, shells, definition['fragment_indices'],
                                  auxbasis=data['auxbasis'], solver=solve_and_save_dimer)
                if old['status'] == 'passed' and cp['dimer_energy_hartree'] > old['dimer_energy_hartree']+5e-8:
                    raise RuntimeError('Dimer retry energy exceeds the retained converged parent energy.')
                monomers = [isolated(m, old['basis']) for m in definition['monomers']]
                energy = cp['dimer_energy_hartree']-sum(m['energy_hartree'] for m in monomers)
                record.update(cp, model_interaction_kcal_mol=energy*HARTREE_TO_KCAL_MOL,
                    isolated_energies_hartree=[m['energy_hartree'] for m in monomers],
                    dataset=definition['dataset'], group=definition['group'], label=definition['label'],
                    scale=definition['scale'], reference_kcal_mol=definition['reference_kcal_mol'],
                    cp_interpretation='frozen dimer basis and common auxiliary metric; separate from raw model energy')
            record['status'] = 'passed'
        except Exception as error:
            record.update(status='failed', error=str(error), traceback=traceback.format_exc())
        record['seconds'] = time.perf_counter()-tick
        report['results'].append(record)
        report.update(passed=sum(r['status'] == 'passed' for r in report['results']),
            failed=sum(r['status'] != 'passed' for r in report['results']),
            elapsed_seconds=time.perf_counter()-started)
        atomic_json(path, report)
        print('END', json.dumps(record, allow_nan=False), flush=True)
        jax.clear_caches()
        if old['key'] in PRIORITY and record['status'] != 'passed':
            report['status'] = 'blocked_by_priority_case'
            atomic_json(path, report)
            raise RuntimeError('Representative repair gate failed; original results preserved.')
    report['status'] = 'complete' if report['passed'] == len(rows) else 'complete_with_failures'
    atomic_json(path, report)
    print('FINISHED', report['status'], report['passed'], '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
