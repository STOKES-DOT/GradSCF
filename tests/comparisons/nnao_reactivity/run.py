"""Native DF-HF curves and R/TS/P energies; checkpoint weights stay frozen."""
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnao_core.native import counterpoise, make_inputs, solve_hf, traditional_shells
from nnao_core.run import FrozenMACE, spin_squared
from nnao_reactivity.protocols import HARTREE_TO_KCAL_MOL, reaction_profile
from gradscf.model.nnao.runtime import atomic_json, file_hash

BASES = ['sto-6g', '6-31g(d,p)', 'def2-svp', 'cc-pvdz', 'cc-pvtz', 'NNAO/MACE']


def conventional_shells(row, basis, basis_assets):
    return [basis_assets[basis+'/'+symbol]['shells'] if basis+'/'+symbol in basis_assets
            else traditional_shells({'symbols': [symbol]}, basis)[0] for symbol in row['symbols']]


def single_point(row, shells, auxbasis):
    data = make_inputs(row, shells, auxbasis=auxbasis)
    state = solve_hf(data['overlap'], data['hcore'], data['factors'],
                     row['nelectron'], row['spin'], data['enuc'])
    return dict(energy_hartree=float(state['energy']), nao=len(data['overlap']),
                orbital_residual=state['orbital_residual'], electron_numbers=state['electron_numbers'],
                cycles=int(state['cycles']), rescue_used=state['rescue_used'],
                spin_squared=spin_squared(state, data['overlap'], row['nelectron'], row['spin']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--profile', choices=['smoke', 'representative'], default='smoke')
    parser.add_argument('--bases', nargs='+')
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('Enable float64 before constructing benchmark arrays.')
    data = json.loads(args.dataset.read_text())
    if file_hash(args.checkpoint) != data['checkpoint_sha256']:
        raise ValueError('Dataset and checkpoint provenance differ.')
    bases = args.bases or (['sto-6g', 'cc-pvtz', 'NNAO/MACE'] if args.profile == 'smoke' else BASES)
    curves, reactions = data['curves'], data['reactions']
    if args.profile == 'smoke':
        curves = [r for r in curves if r['scale'] == 1. and
                  (r['group'] == 'Water-Water' or r['group'] in ['1.01', '1.28'])]
        reactions = [r for r in reactions if r['category'] in ['01', '02', '09']]
    names = sorted({name for r in reactions for name in r['species']})
    jobs = [('curve', row['identifier'], basis, row) for row in curves for basis in bases]
    jobs += [('species', name, basis, data['structures'][name]) for name in names for basis in bases]
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(profile=args.profile, dataset_sha256=file_hash(args.dataset),
        checkpoint_sha256=file_hash(args.checkpoint), snapshot_sha256=file_hash(args.snapshot),
        method='DF-HF', auxbasis=data['auxbasis'], dtype='float64', bases=bases,
        checkpoint_update=data['checkpoint_update'], finite_differences=False, full_eri_storage=False,
        raw_interaction='independent MACE predictions for dimer and isolated monomers',
        cp_control='frozen dimer coefficients in ghost and own calculations; common dimer auxiliary metric',
        source_sha256={str(p): file_hash(p) for p in [Path(__file__),
            Path(__file__).with_name('protocols.py'), Path(__file__).with_name('prepare.py'),
            Path(__file__).parents[1]/'nnao_core/native.py', Path(__file__).parents[1]/'nnao_core/run.py']})
    path = args.output/'configuration.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Resume configuration differs; choose a fresh output directory.')
    atomic_json(path, config)
    path = args.output/'results.json'
    report = json.loads(path.read_text()) if path.exists() else dict(
        status='running', job_id=os.environ.get('SLURM_JOB_ID'), host=socket.gethostname(),
        backend=jax.default_backend(), device=jax.devices()[0].device_kind,
        cpus=os.environ.get('SLURM_CPUS_PER_TASK'), requested_cases=len(jobs),
        curve_points=len(curves), requested_reactions=len(reactions), results=[], isolated_results={})
    predictor = FrozenMACE(args.checkpoint, args.snapshot) if 'NNAO/MACE' in bases else None
    def shells(row, basis):
        return (predictor.shells(row) if basis == 'NNAO/MACE'
                else conventional_shells(row, basis, data.get('basis_assets', {})))

    def isolated(row, basis):
        key = hashlib.sha256((basis+json.dumps(row, sort_keys=True)).encode()).hexdigest()
        if key not in report['isolated_results']:
            result = single_point(row, shells(row, basis), data['auxbasis'])
            report['isolated_results'][key] = dict(result, structure_id=row['structure_id'], basis=basis)
            atomic_json(path, report)
        return report['isolated_results'][key]

    saved = {r['key'] for r in report['results']}
    started = time.perf_counter()
    for task, identifier, basis, row in jobs:
        key = hashlib.sha256(f'{task}|{identifier}|{basis}'.encode()).hexdigest()[:20]
        if key in saved:
            continue
        atomic_json(args.output/'progress.json', dict(task=task, identifier=identifier,
            basis=basis, completed=len(report['results']), total=len(jobs)))
        record = dict(key=key, task=task, identifier=identifier, basis=basis)
        tick = time.perf_counter()
        print('BEGIN', json.dumps(record), flush=True)
        try:
            if task == 'curve':
                dimer = row['structure']
                predicted = shells(dimer, basis)
                atomic_json(args.output/f'{key}-basis.json', dict(row=dimer, atom_shells=predicted))
                cp = counterpoise(dimer, predicted, row['fragment_indices'], auxbasis=data['auxbasis'])
                monomers = [isolated(monomer, basis) for monomer in row['monomers']]
                raw = cp['dimer_energy_hartree']-sum(m['energy_hartree'] for m in monomers)
                record.update(cp, model_interaction_kcal_mol=raw*HARTREE_TO_KCAL_MOL,
                    isolated_energies_hartree=[m['energy_hartree'] for m in monomers],
                    dataset=row['dataset'], group=row['group'], scale=row['scale'], label=row['label'],
                    reference_kcal_mol=row['reference_kcal_mol'],
                    cp_interpretation='fixed-dimer-basis diagnostic; not a CP correction to the independently predicted raw model energy')
            else:
                predicted = shells(row, basis)
                atomic_json(args.output/f'{key}-basis.json', dict(row=row, atom_shells=predicted))
                record.update(single_point(row, predicted, data['auxbasis']), charge=row['charge'],
                              spin=row['spin'], natoms=row['natoms'])
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
    good = {(r['identifier'], r['basis']): r for r in report['results']
            if r['task'] == 'species' and r['status'] == 'passed'}
    report['reactions'] = []
    for reaction in reactions:
        for basis in bases:
            absent = [name for name in reaction['species'] if (name, basis) not in good]
            result = dict(identifier=reaction['identifier'], category=reaction['category'], basis=basis)
            if absent:
                result.update(status='failed', missing_species=absent)
            else:
                energies = {name: good[name, basis]['energy_hartree'] for name in reaction['species']}
                result.update(reaction_profile(reaction['reactants'], reaction['transition'], reaction['products'], energies),
                    status='passed', reference_forward_kcal_mol=reaction['reference_forward_kcal_mol'],
                    reference_reverse_kcal_mol=reaction['reference_reverse_kcal_mol'],
                    reference_reaction_kcal_mol=reaction['reference_reaction_kcal_mol'])
            report['reactions'].append(result)
    report['status'] = 'complete' if report['passed'] == len(jobs) else 'complete_with_failures'
    atomic_json(path, report)
    print('FINISHED', report['status'], report['passed'], '/', len(jobs), flush=True)


if __name__ == '__main__':
    main()
