"""One member of the S/M/L x SZP/DZP/TZP calibration and training campaign.

Separate processes release calibration executables before GPU stage C. Slurm
owns scheduling; this module never allocates GPUs or submits further jobs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from ..initialization import FAMILY_GRID
from ..runtime import atomic_json, file_hash, repository_root, scientific_source_hash


def select_candidate(pilot_dir, references, *, require_gradient_check=True):
    """Select the lowest mean finite-budget free-contraction candidate.

    Require every reference; skipped exponent differences must be explicit. This comparison is a
    local optimization assessment, never a claim of a global contraction limit.
    """
    from ..basis import prepare_direct_basis
    from ..initialization import with_pool_parameters, bind_pool_candidate, core_reference_identity
    p = Path(pilot_dir)
    result = json.loads((p/'summary.json').read_text())
    metadata = json.loads((p/'metadata.json').read_text())
    gate_path=p/'exponent-gradient-check.json'
    gate=json.loads(gate_path.read_text()) if gate_path.exists() else {}
    if (result['status']!='complete' or gate.get('passed') is False
            or (require_gradient_check and gate.get('passed') is not True)):
        raise ValueError('Complete A/B and a passing exponent gradient gate are required.')
    if not require_gradient_check and metadata.get('configuration',{}).get('check_gradients') is not False:
        raise ValueError('The exponent gradient check must be explicitly skipped in pilot metadata.')
    if metadata['references'] != references:
        raise ValueError('A/B references differ from requested basis/reference definition.')
    ids = {r['structure_id'] for r in references['structures']}
    groups = {}
    for row in result['comparisons']:
        groups.setdefault(row['candidate'], []).append(row)
    valid = {}
    for name, rows in groups.items():
        if len(rows) != len(ids) or {r['structure_id'] for r in rows} != ids:
            raise ValueError('Candidate does not cover each reference exactly once.')
        for r in rows:
            if (not np.isfinite([r['free_energy_hartree'], r['pool_energy_hartree']]).all()
                    or r['delta_contraction_hartree'] < -1e-7
                    or not all(r[k].get('converged') and r[k].get('stable')
                               for k in ('pool_scf', 'free_scf'))):
                raise ValueError('Invalid candidate energy, ordering or SCF diagnostics.')
        weights = {r['structure_id']: r['weight'] for r in references['structures']}
        valid[name] = sum(weights[r['structure_id']]*r['free_energy_hartree'] for r in rows)/sum(weights.values())
    if not valid:
        raise ValueError('No evaluated primitive-pool candidate.')
    selected = min(valid, key=valid.get)
    parameters = json.loads((p/f'{selected}-parameters.json').read_text())
    keys = tuple(map(tuple, parameters['keys']))
    channels = {}
    for row in references['structures']:
        layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                                      basis_family=references['basis_family'], core_primitives=references['core_primitives'])
        moved = with_pool_parameters(layout, parameters['values'], keys)
        for atom, l, role, slot, before, after in zip(layout.shell_atoms,
                layout.topology.angular_momenta, layout.roles, layout.slots,
                layout.parameters.exponents, moved.parameters.exponents):
            if slot >= 0:
                key = (layout.symbols[atom], l, role)
                channels[key] = dict(element=key[0], l=l, role=role,
                    reference_exponents_bohr_minus2=np.asarray(before).tolist(),
                    candidate_exponents_bohr_minus2=np.asarray(after).tolist())
    candidate = dict(parameters=parameters, channels=[channels[k] for k in keys],
        selection=dict(candidate=selected, mean_free_energies_hartree=valid,
            exponent_direction_check='passed' if gate.get('passed') is True else 'skipped_by_request',
            basis_family=references['basis_family'], scope='finite-budget local free-contraction assessment',
            summary_sha256=file_hash(p/'summary.json'), optimizer_statuses=[
                r['optimizer_status'] for r in groups[selected]]))
    if core_reference_identity(layout) is not None:
        candidate['core_reference'] = core_reference_identity(layout)
    bind_pool_candidate(layout, candidate)
    return candidate


def run_member(family, references_path, output_dir, *, fixed_candidate=None,
               epochs=1000, calibration_steps=20, contraction_steps=300, starts=3,
               pilot_references=None, calibration_platform='cuda'):
    from .pool_calibration import load_references
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    training = dict(load_references(references_path), basis_family=family)
    pilot = dict(load_references(pilot_references), basis_family=family,
                 core_primitives=training['core_primitives'])
    if 'elements' in training:
        pilot['elements'] = training['elements']
    train_elements = {s for row in training['structures'] for s in row['symbols']}
    pilot_elements = {s for row in pilot['structures'] for s in row['symbols']}
    if not train_elements <= pilot_elements:
        raise ValueError('Provide pilot_references covering every training element before calibrating a new pool.')
    identity = dict(family=family, training_references=training, pilot_references=pilot,
        epochs=epochs, calibration_steps=calibration_steps, contraction_steps=contraction_steps,
        starts=starts, seed=0, learning_rate=1e-4,
        calibration_platform=calibration_platform,
        fixed_candidate=json.loads(Path(fixed_candidate).read_text()) if fixed_candidate else None,
        scientific_source_sha256=scientific_source_hash(repository_root()))
    manifest = root/'campaign.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Campaign configuration/source changed; use a new directory.')
    atomic_json(manifest, identity)
    atomic_json(root/'references.json', training)
    atomic_json(root/'pilot.json', pilot)
    def run(module, arguments, platform):
        environment = dict(os.environ, JAX_PLATFORMS=platform, JAX_ENABLE_X64='1')
        subprocess.run([sys.executable, '-u', '-m', module, *map(str, arguments)],
                       env=environment, check=True)
    prefix = 'gradscf.model.nnao.workflows.'
    summary = root/'calibration/summary.json'
    if not summary.exists() or json.loads(summary.read_text())['status'] != 'complete':
        args = ['--references', root/'pilot.json', '--output-dir', root/'calibration',
                '--calibration-steps', calibration_steps, '--contraction-steps', contraction_steps,
                '--starts', starts, '--seed', 0, '--scf-platform', calibration_platform]
        if fixed_candidate:
            args += ['--fixed-candidate', Path(fixed_candidate).resolve()]
        run(prefix+'pool_calibration', args, 'cuda,cpu' if calibration_platform=='cuda' else 'cpu')
    # Selection stays on CPU; stage C launches a new GPU process.
    run(prefix+'basis_grid', ['select', '--pilot-dir', root/'calibration',
        '--references', root/'pilot.json', '--output-dir', root], 'cpu')
    args = ['--candidate', root/'candidate.json', '--references', root/'references.json',
            '--cache-dir', root/'cache', '--output-dir', root/'training', '--epochs', epochs,
            '--learning-rate', 1e-4, '--seed', 0, '--storage', 'mmap', '--platform', 'cuda']
    if (root/'training/last.npz').exists():
        args.append('--resume')
    run(prefix+'fixed_pool', args, 'cuda,cpu')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run')
    run.add_argument('--family', choices=tuple(FAMILY_GRID), required=True)
    run.add_argument('--references', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--fixed-candidate', type=Path)
    run.add_argument('--pilot-references', type=Path)
    run.add_argument('--epochs', type=int, default=1000)
    run.add_argument('--calibration-steps', type=int, default=20)
    run.add_argument('--contraction-steps', type=int, default=300)
    run.add_argument('--starts', type=int, default=3)
    run.add_argument('--calibration-platform', choices=('cpu','cuda'), default='cuda')
    select = sub.add_parser('select')
    select.add_argument('--pilot-dir', type=Path, required=True)
    select.add_argument('--references', type=Path, required=True)
    select.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'select':
        candidate = select_candidate(args.pilot_dir, json.loads(args.references.read_text()))
        atomic_json(args.output_dir/'candidate.json', candidate)
    else:
        run_member(args.family, args.references, args.output_dir,
            fixed_candidate=args.fixed_candidate, epochs=args.epochs,
            calibration_steps=args.calibration_steps, contraction_steps=args.contraction_steps,
            starts=args.starts, pilot_references=args.pilot_references,
            calibration_platform=args.calibration_platform)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
