"""RHF C-C stretch; adapted from the frozen torsion runner, no solver changes."""
from pathlib import Path
import json
import os
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, file_hash, repository_root, scientific_source_hash
from gradscf.model.nnao.training.trainer import make_mace_predictor

ROOT = Path(__file__).resolve().parent
OUT = ROOT
PARENT = ROOT.parent / 'nnao-relative500-20261018'
IDENTIFIER = None


def main():
    assert jax.config.jax_enable_x64 and jax.default_backend() == 'gpu'
    snapshot = json.loads((PARENT / 'snapshot.json').read_text())
    assert file_hash(PARENT / 'checkpoint-current.npz') == snapshot['checkpoint_sha256']
    assert scientific_source_hash(repository_root()) == snapshot['source_sha256']
    with np.load(PARENT / 'checkpoint-current.npz', allow_pickle=False) as saved:
        identity = json.loads(saved['metadata'].item())['identity']
        weights = np.array(saved['parameters'])
    from geometry import stretch_rows
    training = identity['references']['structures']
    base = next(r for r in training if r['structure_id'] == 'GMTKN55-v2/ADIM6/AM2')
    row = json.loads((OUT/'structure.json').read_text())
    expected = stretch_rows(base, [row['distance_angstrom']])[0]
    np.testing.assert_allclose(row['coords_angstrom'], expected['coords_angstrom'], atol=1e-12, rtol=0)
    assert row['structure_id'] == IDENTIFIER
    atomic_json(OUT / 'structure.json', row)
    configuration = dict(structure_id=IDENTIFIER, source_geometry_in_training_set=False, distance_angstrom=row['distance_angstrom'], training_count=500, checkpoint=snapshot, starts=['mace', 'reference'],
        script_sha256=file_hash(__file__), fixed_exponents=True, fixed_core=True,
        method='DF-RHF', auxbasis=identity['auxbasis'], spherical=True,
        implicit_tolerance=1e-8, conv_tol_energy=1e-12, conv_tol_density=1e-10,
        conv_tol_grad=1e-9, optimizer='SciPy L-BFGS-B with analytic implicit SCF gradient',
        maxiter=1000, maxfun=3000, maxls=30, ftol=1e-15, gtol=1e-7,
        reported_gradient_norm_tolerance=1e-5, finite_differences=False,
        coefficient_variables='Independent per-atom radial coefficients; shared across m components')
    atomic_json(OUT / 'configuration.json', configuration)
    cpu, gpu = jax.devices('cpu')[0], jax.devices('gpu')[0]
    report = dict(status='running', configuration=configuration, device=gpu.device_kind,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
        starts=[], evaluations=0)
    started = time.perf_counter()

    def progress(phase, **kwargs):
        report.update(phase=phase, elapsed_seconds=time.perf_counter()-started, **kwargs)
        atomic_json(OUT / 'progress.json', report)

    progress('prepare_layout_and_cache')
    with jax.default_device(cpu):
        layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
            basis_family=identity['references']['basis_family'],
            core_primitives=identity['references']['core_primitives'], cart=False)
        layout = bind_pool_candidate(layout, identity['candidate'])
    (OUT / 'cache').mkdir(exist_ok=True)
    tick = time.perf_counter()
    free, manifest = prepare_fixed_pool(row, layout, OUT / 'cache' / 'pool.npz',
        storage='device', auxbasis=identity['auxbasis'], implicit_tolerance=1e-8)
    report.update(cache_seconds=time.perf_counter()-tick, cache_bytes=manifest['bytes'],
                  nao=layout.topology.nao, pool_nao=manifest['pool_nao'])
    _, predict, _, _ = make_mace_predictor(identity['seed'], identity['model_config'], return_mask=True)
    graph = build_graph(row['atomic_numbers'], row['coords_angstrom'],
        element_order=identity['model_config']['elements'], cutoff=identity['model_config']['cutoff'])
    raw_mace = np.asarray(predict(jnp.asarray(weights), graph))
    raw_reference = np.asarray(layout.reference_outputs())
    mask = np.zeros_like(raw_mace, dtype=bool)
    shell_spec = []
    for atom, slot, count, columns, ell, role in zip(layout.shell_atoms, layout.slots,
            layout.topology.primitive_counts, layout.topology.contraction_counts,
            layout.topology.angular_momenta, layout.roles):
        shell_spec.append(dict(atom=int(atom), slot=int(slot), primitive_count=int(count),
                               contraction_count=int(columns), angular_momentum=int(ell), role=role))
        if slot >= 0:
            mask[atom, slot:slot+columns, :count] = True

    def normalized(c):
        c = np.array(c, copy=True)
        for shell in shell_spec:
            atom, slot, count = shell['atom'], shell['slot'], shell['primitive_count']
            if slot >= 0:
                for column in range(slot, slot+shell['contraction_count']):
                    c[atom, column, :count] /= np.linalg.norm(c[atom, column, :count])
        return c

    mace, reference = map(normalized, (raw_mace, raw_reference))
    report.update(active_raw_coefficients=int(mask.sum()), shells=shell_spec)
    exports = []

    def export(c, label, energy):
        with jax.default_device(cpu):
            bound = layout.bind(jax.device_put(c, cpu))
            shells = layout.atom_shells(bound)
        exports[:] = [record for record in exports if record['basis'] != label]
        exports.append(dict(structure_id=IDENTIFIER, basis=label, symbols=row['symbols'],
            coords_angstrom=row['coords_angstrom'], atom_labels=[f'{s}{i}' for i,s in enumerate(row['symbols'])],
            atom_shells=shells, nao=layout.topology.nao, native_energy_hartree=energy))
        (OUT / 'frozen-bases.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in exports))

    timings = []
    def evaluate(c):
        tick = time.perf_counter()
        energy, derivative, details = free.evaluate_device(jax.device_put(c, gpu), gradient=True)
        derivative = np.asarray(derivative)
        assert np.max(np.abs(derivative[~mask]), initial=0.) < 1e-10
        timings.append(time.perf_counter()-tick)
        report['evaluations'] += 1
        return energy, derivative, details

    export(mace, 'MACE-update3351', None)
    progress('mace_implicit_forward_backward')
    e0, g0, info0 = evaluate(mace)
    report.update(mace_energy_hartree=e0, mace_gradient_norm=float(np.linalg.norm(g0)),
                  mace_scf=info0)
    export(mace, 'MACE-update3351', e0)
    er, gr, infor = evaluate(reference)
    report.update(reference_energy_hartree=er)
    candidates = [(e0, 'original_mace', mace), (er, 'original_reference', reference)]
    for name, initial, initial_e, initial_g in [('mace', mace, e0, g0), ('reference', reference, er, gr)]:
        tick = time.perf_counter()
        start_evaluations = report['evaluations']
        history = [dict(iteration=0, energy_hartree=initial_e, gradient_norm=float(np.linalg.norm(initial_g)))]
        current = dict(x=initial[mask].copy(), energy=initial_e, gradient=initial_g[mask].copy())
        template = initial.copy()
        best = [initial_e, initial.copy()]

        def unpack(x):
            c = template.copy()
            c[mask] = x
            return c

        def fun(x):
            if not np.array_equal(x, current['x']):
                value, derivative, details = evaluate(unpack(x))
                current.update(x=np.array(x, copy=True), energy=value, gradient=derivative[mask], scf=details)
            return current['energy'], current['gradient']

        def callback(x):
            energy, gradient = fun(x)
            history.append(dict(iteration=len(history), energy_hartree=energy,
                                gradient_norm=float(np.linalg.norm(gradient))))
            if energy < best[0]:
                best[:] = [energy, unpack(x)]
            if len(history) % 10 == 0:
                np.savez(OUT / f'{name}-accepted.npz', coefficients=unpack(x), gradient=gradient)
                atomic_json(OUT / f'{name}-history.json', history)
                progress(name, active_iteration=len(history)-1, active_energy_hartree=energy,
                         active_gradient_norm=float(np.linalg.norm(gradient)))
                print(name, len(history)-1, energy, float(np.linalg.norm(gradient)), flush=True)

        progress(name, active_iteration=0)
        try:
            result = minimize(fun, initial[mask], method='L-BFGS-B', jac=True, callback=callback,
                options=dict(maxiter=1000, maxfun=3000, maxls=30, maxcor=20, ftol=1e-15, gtol=1e-7))
            energy, gradient = fun(result.x)
            if energy < best[0]:
                best[:] = [energy, unpack(result.x)]
            entry = dict(start=name, status='complete', optimizer_success=bool(result.success),
                optimizer_message=str(result.message), iterations=int(result.nit), function_calls=int(result.nfev),
                initial_energy_hartree=initial_e, energy_hartree=energy,
                gradient_norm=float(np.linalg.norm(gradient)),
                gradient_converged=float(np.linalg.norm(gradient)) <= 1e-5,
                best_accepted_energy_hartree=best[0])
            np.savez(OUT / f'{name}-final.npz', coefficients=unpack(result.x), gradient=gradient)
        except Exception as exc:
            entry = dict(start=name, status='failed', error=str(exc), traceback=traceback.format_exc(),
                         best_accepted_energy_hartree=best[0])
        entry.update(seconds=time.perf_counter()-tick, evaluations=report['evaluations']-start_evaluations)
        atomic_json(OUT / f'{name}-history.json', history)
        np.savez(OUT / f'{name}-best.npz', coefficients=best[1])
        candidates.append((best[0], name, best[1]))
        report['starts'].append(entry)
        progress(name+'_finished')
        print('END', entry, flush=True)

    energy, name, coefficients = min(candidates, key=lambda r: r[0])
    checked, gradient, details = evaluate(coefficients)
    np.testing.assert_allclose(checked, energy, atol=1e-9, rtol=0.)
    np.savez(OUT / 'best-coefficients.npz', coefficients=coefficients, gradient=gradient,
             mace_coefficients=mace, reference_coefficients=reference, active_mask=mask)
    export(coefficients, 'free-contraction-best', checked)
    good = [r for r in report['starts'] if r['status'] == 'complete']
    report.update(best_start=name, best_energy_hartree=checked, best_gradient_norm=float(np.linalg.norm(gradient)),
        best_gradient_converged=float(np.linalg.norm(gradient)) <= 1e-5, best_scf=details,
        mace_minus_free_hartree=e0-checked, mace_minus_free_millihartree=1000*(e0-checked),
        mace_minus_free_kcal_mol=627.5094740631*(e0-checked),
        endpoint_spread_hartree=max(r['energy_hartree'] for r in good)-min(r['energy_hartree'] for r in good) if good else None,
        first_forward_backward_seconds=timings[0], median_hot_forward_backward_seconds=float(np.median(timings[1:])),
        status='complete' if len(good) == 2 else 'complete_with_failures',
        interpretation='Finite-budget per-structure contraction optimization; no global optimality claim.')
    progress('complete')
    atomic_json(OUT / 'summary.json', report)
    assert file_hash(PARENT / 'checkpoint-current.npz') == snapshot['checkpoint_sha256']
    print('SUMMARY', json.dumps({k:v for k,v in report.items() if k not in {'configuration','shells','mace_scf','best_scf'}}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
