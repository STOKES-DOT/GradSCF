"""Native, implicit-gradient contraction repair with explicit RHF branch search."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
os.environ['GRADSCF_NATIVE_THREADS'] = '4'
from pathlib import Path
import argparse
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from probe_native import prepare, atomic_seed, OUT, SOURCE
from gradscf.model.nnao.runtime import atomic_json
from gradscf.model.nnao.training.objective import BasisRHF
from gradscf.model.nnao.training.initial_guess import atomic_guess_data


def normalize(c, layout):
    c = np.array(c, copy=True)
    for atom, slot, count, columns in zip(layout.shell_atoms, layout.slots,
            layout.topology.primitive_counts, layout.topology.contraction_counts):
        if slot >= 0:
            for column in range(slot, slot + columns):
                c[atom, column, :count] /= np.linalg.norm(c[atom, column, :count])
    return c


def evaluator(experiment, layout):
    guess_data = atomic_guess_data(layout)
    count = [0]

    def evaluate(c, gradient=True):
        c = jnp.asarray(c)
        physical = experiment.project(c, *experiment.raw)
        seed = atomic_seed(experiment, layout, c)
        candidates, failures = [], {}
        for name, options in (('hcore', {}), ('atomic', dict(init_density=seed))):
            try:
                energy, info = experiment.value(physical, experiment.enuc, guess_data, **options)
                checked = BasisRHF._checked_info(energy, info)
                candidates.append((float(energy), name, options, checked))
            except Exception as error:
                failures[name] = str(error)
        if not candidates:
            raise RuntimeError(f'No valid restricted SCF branch: {failures}')
        energy, name, options, info = min(candidates, key=lambda row: row[0])
        derivative = None
        if gradient:
            (checked_energy, checked_info), cotangent = experiment.gradient(
                physical, experiment.enuc, guess_data, **options)
            np.testing.assert_allclose(checked_energy, energy, atol=1e-9, rtol=0)
            info = BasisRHF._checked_info(checked_energy, checked_info)
            _, pullback = jax.vjp(lambda x: experiment.project(x, *experiment.raw), c)
            derivative = np.asarray(pullback(cotangent)[0])
            if not np.isfinite(derivative).all():
                raise RuntimeError('Nonfinite implicit coefficient gradient')
        count[0] += 1
        return energy, derivative, dict(info, selected_guess=name,
            branch_energies={r[1]: r[0] for r in candidates}, failed_guesses=failures)

    return evaluate, count


def optimize(evaluate, initial, mask, layout, directory, name):
    initial = normalize(initial, layout)
    template = initial.copy()
    current = dict(x=None)
    history = []
    started = time.perf_counter()

    def unpack(x):
        c = template.copy()
        c[mask] = x
        return c

    def fun(x):
        if current['x'] is None or not np.array_equal(current['x'], x):
            energy, gradient, info = evaluate(unpack(x))
            current.update(x=np.array(x, copy=True), energy=energy, gradient=gradient[mask], info=info)
        return current['energy'], current['gradient']

    def callback(x):
        energy, gradient = fun(x)
        history.append(dict(iteration=len(history)+1, energy_hartree=energy,
            gradient_norm=float(np.linalg.norm(gradient)), selected_guess=current['info']['selected_guess']))
        if len(history) % 10 == 0:
            atomic_json(directory / f'{name}-history.json', history)
            print('ITERATION', directory.name, name, history[-1], flush=True)

    result = minimize(fun, initial[mask], jac=True, method='L-BFGS-B', callback=callback,
        options=dict(maxiter=1000, maxfun=4000, maxls=40, maxcor=20, ftol=1e-15, gtol=1e-7))
    c = normalize(unpack(result.x), layout)
    energy, gradient, info = evaluate(c)
    norm = float(np.linalg.norm(gradient[mask]))
    entry = dict(start=name, energy_hartree=energy, canonical_gradient_norm=norm,
        gradient_converged=norm <= 1e-5, optimizer_success=bool(result.success),
        optimizer_message=str(result.message), iterations=int(result.nit),
        function_calls=int(result.nfev), seconds=time.perf_counter()-started, scf=info)
    np.savez(directory / f'{name}-final.npz', coefficients=c, gradient=gradient)
    atomic_json(directory / f'{name}-history.json', history)
    print('END_START', directory.name, entry, flush=True)
    return entry, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--last-index', type=int, default=55)
    args = parser.parse_args()
    rows = json.loads((SOURCE / 'structures.json').read_text())
    summary_path = OUT / 'repaired-nnao.json'
    report = json.loads(summary_path.read_text()) if summary_path.exists() else dict(
        status='running', results=[], method='native DF-RHF', auxbasis='def2-universal-jkfit',
        fixed_exponents=True, fixed_core=True, branch_search=['hcore', 'atomic'],
        gradient='existing implicit SCF physical-input VJP + contraction pullback',
        finite_differences=False, jax=jax.__version__, backend=jax.default_backend())
    done = {r['index'] for r in report['results']}
    previous = np.load(SOURCE / 'distance-023/best-coefficients.npz')['coefficients']
    for index in range(24, args.last_index + 1):
        folder = OUT / f'distance-{index:03d}'
        if index in done:
            previous = np.load(folder / 'best-coefficients.npz')['coefficients']
            continue
        folder.mkdir(exist_ok=True)
        row, layout, old, experiment, manifest = prepare(index)
        evaluate, evaluations = evaluator(experiment, layout)
        mask = np.asarray(old['active_mask'])
        mace_energy, _, mace_info = evaluate(old['mace_coefficients'], gradient=False)
        print('BEGIN', index, row['distance_angstrom'], 'MACE', mace_energy, flush=True)
        solutions = [optimize(evaluate, seed, mask, layout, folder, name)
                     for name, seed in (('previous', previous), ('archived', old['coefficients']))]
        accepted = [(entry, c) for entry, c in solutions if entry['gradient_converged']]
        if not accepted:
            atomic_json(folder / 'failure.json', dict(index=index, starts=[r[0] for r in solutions]))
            raise RuntimeError(f'Neither start reached the canonical gradient tolerance: {index}')
        best, coefficients = min(accepted, key=lambda pair: pair[0]['energy_hartree'])
        assert best['energy_hartree'] <= mace_energy + 1e-8
        np.savez(folder / 'best-coefficients.npz', coefficients=coefficients,
                 mace_coefficients=old['mace_coefficients'], active_mask=mask)
        frozen = []
        for label, c, energy in (('MACE-update3351', old['mace_coefficients'], mace_energy),
                                ('free-contraction-best', coefficients, best['energy_hartree'])):
            frozen.append(dict(structure_id=row['structure_id'], basis=label,
                symbols=row['symbols'], coords_angstrom=row['coords_angstrom'],
                atom_labels=[f'{s}{i}' for i, s in enumerate(row['symbols'])],
                atom_shells=layout.atom_shells(layout.bind(jnp.asarray(c))), nao=layout.topology.nao,
                native_energy_hartree=energy))
        (folder / 'frozen-bases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in frozen))
        result = dict(index=index, distance_angstrom=row['distance_angstrom'], status='passed',
            mace_energy_hartree=mace_energy, mace_scf=mace_info, free_energy_hartree=best['energy_hartree'],
            free_scf=best['scf'], gradient_norm=best['canonical_gradient_norm'],
            starts=[r[0] for r in solutions], evaluations=evaluations[0],
            endpoint_spread_hartree=max(r[0]['energy_hartree'] for r in solutions)-min(r[0]['energy_hartree'] for r in solutions))
        atomic_json(folder / 'result.json', result)
        report['results'].append(result)
        report['status'] = 'complete' if index == 55 else 'running'
        atomic_json(summary_path, report)
        previous = coefficients
        print('COMPLETE_POINT', index, result['free_energy_hartree'], result['gradient_norm'], flush=True)


if __name__ == '__main__':
    main()
