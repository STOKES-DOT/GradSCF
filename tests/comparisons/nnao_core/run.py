"""Native HF core benchmarks against fixed traditional and frozen MACE bases."""
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
from scipy.integrate import simpson

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnao_core.native import counterpoise, make_inputs, solve_hf, static_polarizability, traditional_shells
from nnao_core.protocols import atomic_radial_density
from gradscf.model.nnao.runtime import atomic_json, file_hash, repository_root, scientific_source_hash

BASES = ['sto-6g', 'STO-6G+P', '6-31g(d,p)', 'def2-svp', 'cc-pvdz', 'cc-pvtz', 'ma-def2-qzvpp', 'NNAO/MACE']


class FrozenMACE:
    def __init__(self, checkpoint, snapshot):
        from gradscf.model.nnao.training.trainer import make_mace_predictor
        self.snapshot = json.loads(snapshot.read_text())
        if file_hash(checkpoint) != self.snapshot['checkpoint_sha256']:
            raise ValueError('Frozen MACE checkpoint checksum differs.')
        actual = scientific_source_hash(repository_root())
        if actual != self.snapshot['source_sha256']:
            raise ValueError(f'Frozen scientific source differs: {actual}.')
        with np.load(checkpoint, allow_pickle=False) as saved:
            self.identity = json.loads(saved['metadata'].item())['identity']
            self.parameters = jnp.asarray(np.array(saved['parameters']))
        _, self.predict, _, _ = make_mace_predictor(self.identity['seed'],
                                                   self.identity['model_config'], return_mask=True)
        self.cache = {}

    def shells(self, row):
        from gradscf.model.nnao.basis import prepare_direct_basis
        from gradscf.model.nnao.graph import build_graph
        from gradscf.model.nnao.initialization import bind_pool_candidate
        key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if key not in self.cache:
            cfg = self.identity['model_config']
            graph = build_graph(row['atomic_numbers'], row['coords_angstrom'],
                element_order=cfg['elements'], cutoff=cfg['cutoff'],
                total_charge=row['charge'], spin=row['spin'])
            raw = np.asarray(self.predict(self.parameters, graph))
            if not np.isfinite(raw).all():
                raise ValueError('MACE produced nonfinite basis coefficients.')
            with jax.default_device(jax.devices('cpu')[0]):
                refs = self.identity['references']
                layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                    charge=row['charge'], spin=row['spin'], cart=False,
                    basis_family=refs['basis_family'], core_primitives=refs['core_primitives'])
                layout = bind_pool_candidate(layout, self.identity['candidate'])
                shells = layout.atom_shells(layout.bind(jnp.asarray(raw)))
            self.cache[key] = shells
        return self.cache[key]


def spin_squared(state, overlap, nelectron, spin):
    if spin == 0:
        return 0.
    na, nb = (nelectron+spin)//2, (nelectron-spin)//2
    ca, cb = state['mo'][0, :, :na], state['mo'][1, :, :nb]
    return float((spin/2)*(spin/2+1)+nb-np.sum(abs(ca.T@overlap@cb)**2))


def atomic_case(row, shells, output, key, auxbasis):
    data = make_inputs(row, shells, auxbasis=auxbasis)
    state = solve_hf(data['overlap'], data['hcore'], data['factors'],
                     row['nelectron'], row['spin'], data['enuc'])
    radius = np.r_[0., np.geomspace(1e-7, 60., 6000)]
    radial = atomic_radial_density(shells[0], state['density'], radius)
    number = float(simpson(radial, x=radius))
    np.testing.assert_allclose(number, row['nelectron'], atol=1e-5, rtol=0)
    mean_radius = float(simpson(radius*radial, x=radius)/row['nelectron'])
    np.savez_compressed(output/f'{key}-radial.npz', radius_bohr=radius,
        radial_density_electrons_per_bohr=radial,
        normalized_radial_density=radial/row['nelectron'], density_matrix=state['density'])
    return dict(energy_hartree=float(state['energy']), nao=len(data['overlap']),
        spin_squared=spin_squared(state, data['overlap'], row['nelectron'], row['spin']),
        radial_electron_number=number, mean_radius_bohr=mean_radius,
        orbital_residual=state['orbital_residual'], electron_numbers=state['electron_numbers'],
        radial_file=f'{key}-radial.npz',
        neural_domain='charge/spin/isolated-atom extrapolation from neutral closed-shell molecular training')


def polar_case(row, shells, auxbasis):
    data = make_inputs(row, shells, auxbasis=auxbasis)
    state = solve_hf(data['overlap'], data['hcore'], data['factors'], row['nelectron'], 0, data['enuc'])
    return dict(static_polarizability(data, state, row['nelectron']),
        energy_hartree=float(state['energy']), nao=len(data['overlap']),
        orbital_residual=state['orbital_residual'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--profile', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--tasks', default='atoms,bsse,polarizability')
    parser.add_argument('--bases', nargs='+', default=None)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('Enable JAX float64 before running numerical cases.')
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(args.dataset.read_text())
    tasks = args.tasks.split(',')
    if not set(tasks) <= {'atoms', 'bsse', 'polarizability'}:
        raise ValueError('Unknown core benchmark task.')
    bases = args.bases or (BASES if args.profile == 'full'
                          else ['STO-6G+P', 'def2-svp', 'NNAO/MACE'])
    groups = dict(atoms=dataset['atoms'], bsse=dataset['s22'], polarizability=dataset['polarizability'])
    if args.profile == 'smoke':
        groups['atoms'] = [row for row in groups['atoms'] if row['label'] in ['H', 'H-']]
        groups['bsse'] = [row for row in groups['bsse'] if row['structure_id'] == 'S22/02']
        groups['polarizability'] = [row for row in groups['polarizability'] if row['label'] == 'ethylene']
    jobs = [(task, row, basis) for task in tasks for row in groups[task] for basis in bases]
    if args.profile == 'full' and 'atoms' in tasks and 'aug-cc-pv5z' not in bases:
        jobs += [('atoms', row, 'aug-cc-pv5z') for row in groups['atoms']]
    config = dict(profile=args.profile, dataset_sha256=file_hash(args.dataset),
        checkpoint_sha256=file_hash(args.checkpoint), bases=bases, tasks=tasks,
        method='HF', auxbasis=dataset['auxbasis'], dtype='float64',
        mace_checkpoint_update=3351, raw_eri_storage=False, finite_differences=False,
        benchmark_sources_sha256={name: file_hash(Path(__file__).with_name(name))
                                 for name in ['run.py', 'native.py', 'protocols.py', 'prepare.py']})
    manifest = args.output/'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Existing benchmark configuration changed; use a new output directory.')
    atomic_json(manifest, config)
    if (args.output/'results.json').exists():
        report = json.loads((args.output/'results.json').read_text())
    else:
        report = dict(status='running', job_id=os.environ.get('SLURM_JOB_ID'),
            hostname=socket.gethostname(), backend=jax.default_backend(),
            device=jax.devices()[0].device_kind, cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
            requested_cases=len(jobs), results=[])
    saved = {row['key'] for row in report['results']}
    predictor = FrozenMACE(args.checkpoint, args.snapshot) if 'NNAO/MACE' in bases else None
    started = time.perf_counter()
    for task, row, basis in jobs:
        key = hashlib.sha256(f"{task}|{row['structure_id']}|{basis}".encode()).hexdigest()[:20]
        if key in saved:
            continue
        tick = time.perf_counter()
        atomic_json(args.output/'progress.json', dict(task=task, structure_id=row['structure_id'], basis=basis,
            completed=len(report['results']), total=len(jobs), elapsed_seconds=time.perf_counter()-started))
        record = dict(key=key, task=task, structure_id=row['structure_id'],
                      label=row.get('label', row['structure_id']), basis=basis)
        print('BEGIN', json.dumps(record), flush=True)
        try:
            shells = predictor.shells(row) if basis == 'NNAO/MACE' else traditional_shells(row, basis)
            atomic_json(args.output/f'{key}-basis.json', dict(row=row, atom_shells=shells))
            if task == 'atoms':
                values = atomic_case(row, shells, args.output, key, dataset['auxbasis'])
            elif task == 'bsse':
                values = counterpoise(row, shells, row['fragment_indices'], auxbasis=dataset['auxbasis'])
                values['bsse_percent_reference_binding'] = 100*values['bsse_kcal_mol']/row['reference_dissociation_kcal_mol']
            else:
                values = polar_case(row, shells, dataset['auxbasis'])
            record.update(values, status='passed')
        except Exception as error:
            record.update(status='failed', error=str(error), traceback=traceback.format_exc())
        record['seconds'] = time.perf_counter()-tick
        report['results'].append(record)
        report.update(passed=sum(r['status'] == 'passed' for r in report['results']),
            failed=sum(r['status'] != 'passed' for r in report['results']), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/'results.json', report)
        print('END', json.dumps(record, allow_nan=False), flush=True)
        jax.clear_caches()
    report['status'] = 'complete' if report['passed'] == len(jobs) else 'complete_with_failures'
    atomic_json(args.output/'results.json', report)
    print('FINISHED', report['status'], report['passed'], '/', len(jobs), flush=True)


if __name__ == '__main__':
    main()
