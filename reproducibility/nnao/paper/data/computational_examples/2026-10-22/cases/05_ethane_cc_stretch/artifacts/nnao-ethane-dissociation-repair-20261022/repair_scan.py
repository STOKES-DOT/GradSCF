"""Fused implicit-SCF continuation; audit alternative seeds at every endpoint."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from reoptimize import prepare, evaluator, optimize, normalize, OUT, SOURCE, atomic_seed
from gradscf.model.nnao.runtime import atomic_json, scientific_source_hash, repository_root

RESULTS = OUT / 'native-rerun'


def branch_evaluator(experiment, layout, branch):
    def evaluate(c, gradient=True):
        options = dict(init_density=atomic_seed(experiment, layout, c)) if branch == 'atomic' else {}
        energy, derivative, info = experiment.evaluate_device(jnp.asarray(c), gradient=gradient, **options)
        return energy, None if derivative is None else np.asarray(derivative), dict(info, selected_guess=branch)
    return evaluate


def main():
    RESULTS.mkdir(exist_ok=True)
    target = RESULTS / 'scan-summary.json'
    report = json.loads(target.read_text()) if target.exists() else dict(status='running', results=[],
        method='native DF-RHF', fixed_exponents=True, fixed_core=True, finite_differences=False,
        implicit_tolerance=1e-8, required_canonical_gradient_norm=1e-5,
        branch_policy='hcore/atomic at start and endpoint; selected seed inside each optimization',
        inner_evaluation='one fused SCF forward and implicit backward',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scientific_source_sha256=scientific_source_hash(repository_root()), jax=jax.__version__, backend=jax.default_backend())
    done = {r['index'] for r in report['results']}
    previous = np.load(SOURCE / 'distance-023/best-coefficients.npz')['coefficients']
    started = time.perf_counter()
    for index in range(24, 56):
        folder = RESULTS / f'distance-{index:03d}'
        if index in done:
            previous = np.load(folder / 'best-coefficients.npz')['coefficients']
            continue
        folder.mkdir(exist_ok=True)
        row, layout, old, experiment, manifest = prepare(index)
        audit, evaluations = evaluator(experiment, layout)
        mask = np.asarray(old['active_mask'])
        mace_energy, _, mace_info = audit(old['mace_coefficients'], gradient=False)
        print('BEGIN', index, row['distance_angstrom'], mace_energy, flush=True)
        starts, accepted = [], []
        for name, initial in (('previous', previous), ('archived', old['coefficients'])):
            c = normalize(initial, layout)
            _, _, initial_info = audit(c, gradient=False)
            branch = initial_info['selected_guess']
            for attempt in range(3):
                evaluate = branch_evaluator(experiment, layout, branch)
                entry, c = optimize(evaluate, c, mask, layout, folder, f'{name}-{attempt}-{branch}')
                checked, _, info = audit(c, gradient=False)
                entry.update(endpoint_audit=info, endpoint_lowering_hartree=entry['energy_hartree']-checked)
                starts.append(entry)
                if entry['energy_hartree']-checked > 1e-8:
                    branch = info['selected_guess']
                    continue
                if entry['gradient_converged']:
                    accepted.append((entry, c))
                break
        if not accepted:
            atomic_json(folder / 'failure.json', dict(index=index, starts=starts))
            raise RuntimeError(f'No converged audited endpoint: {index}')
        best, coefficients = min(accepted, key=lambda pair: pair[0]['energy_hartree'])
        assert best['energy_hartree'] <= mace_energy + 1e-8
        np.savez(folder / 'best-coefficients.npz', coefficients=coefficients,
                 mace_coefficients=old['mace_coefficients'], active_mask=mask)
        frozen = []
        for label, c, energy in (('MACE-update3351', old['mace_coefficients'], mace_energy),
                                ('free-contraction-best', coefficients, best['energy_hartree'])):
            frozen.append(dict(structure_id=row['structure_id'], basis=label, symbols=row['symbols'],
                coords_angstrom=row['coords_angstrom'], atom_labels=[f'{s}{i}' for i, s in enumerate(row['symbols'])],
                atom_shells=layout.atom_shells(layout.bind(jnp.asarray(c))), nao=layout.topology.nao,
                native_energy_hartree=energy))
        (folder / 'frozen-bases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in frozen))
        result = dict(index=index, distance_angstrom=row['distance_angstrom'], status='passed',
            mace_energy_hartree=mace_energy, mace_scf=mace_info, free_energy_hartree=best['energy_hartree'],
            free_scf=best['endpoint_audit'], gradient_norm=best['canonical_gradient_norm'], starts=starts,
            valid_endpoint_count=len(accepted),
            endpoint_spread_hartree=max(r[0]['energy_hartree'] for r in accepted)-min(r[0]['energy_hartree'] for r in accepted),
            elapsed_seconds=time.perf_counter()-started)
        atomic_json(folder / 'result.json', result)
        report['results'].append(result)
        report['status'] = 'complete' if index == 55 else 'running'
        atomic_json(target, report)
        previous = coefficients
        print('COMPLETE_POINT', index, result['free_energy_hartree'], result['gradient_norm'], flush=True)


if __name__ == '__main__':
    main()
