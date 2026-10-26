"""Six controlled BSE kernels; real saved water input plus synthetic scale points.

No SCF/GW timing is included. Synthetic inputs are declared numerical spectra
and factors, not calculated QP levels or a molecular scaling series. This probe
does not change the production solver default or infer a universal crossover.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('screening', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case-index', type=int, choices=(0, 1, 2))
    parser.add_argument('--solver', choices=('dense', 'davidson'))
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    from gradscf.bse import BSEConfig, make_bse_space, run_bse

    def inputs(index):
        if index == 0:
            saved = np.load(args.screening, allow_pickle=False)
            return ('water-saved', jnp.asarray(saved['qp_energy']), jnp.asarray(saved['screening_energy']),
                    jnp.asarray(saved['mo_factors']), int(saved['nocc']), 1, True)
        nocc, nvir, tda = ((5, 100, True), (10, 200, False))[index-1]
        energy = jnp.concatenate((jnp.linspace(-1.5, -.4, nocc), jnp.linspace(.2, 3., nvir)))
        raw = np.random.default_rng(127+nvir).normal(size=(8, energy.size, energy.size))*.005
        factors = jnp.asarray((raw+raw.swapaxes(1, 2))*.5)
        return 'synthetic', energy, energy, factors, nocc, 5, tda
    report = dict(scope='shared-input BSE kernel, not independent geometry workflow',
        device='cpu', threads=1, dtype='float64', seed_rule='127+nvir',
        production_default_changed=False, universal_threshold_inferred=False,
        isolated_case=args.case_index is not None and args.solver is not None, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    for index in range(3) if args.case_index is None else (args.case_index,):
        origin, qp, screen, factors, nocc, roots, tda = inputs(index)
        space = make_bse_space(qp.size, nocc)
        dimension = nocc*(qp.size-nocc)
        reference = None
        for solver in ('dense', 'davidson') if args.solver is None else (args.solver,):
            cfg = BSEConfig(nroots=roots, tda=tda, solver=solver, conv_tol=1e-8,
                max_dense=dimension, max_cycle=100, max_space=40)
            numerical = jax.jit(lambda q, e, l: run_bse(q, e, l, space, config=cfg))
            row = dict(origin=origin, dimension=int(dimension), naux=int(factors.shape[0]),
                nroots=roots, tda=tda, solver=solver, max_dense=dimension,
                cache='fixed numerical inputs, reusable executable, no cached result', samples=[])
            try:
                for repeat in range(4):
                    tick = time.perf_counter()
                    result = jax.block_until_ready(numerical(qp, screen, factors))
                    row['samples'].append(dict(repeat=repeat, seconds=time.perf_counter()-tick,
                        cold_or_warm='first' if repeat == 0 else 'warm'))
                    assert np.all(result.converged) and np.isfinite(result.excitation_energies).all()
                if reference is None:
                    reference = np.asarray(result.excitation_energies)
                else:
                    np.testing.assert_allclose(result.excitation_energies, reference, atol=2e-8, rtol=2e-8)
                row.update(status='ok', energies_hartree=np.asarray(result.excitation_energies).tolist(),
                           residual_norms=np.asarray(result.residual_norms).tolist())
            except Exception as error:
                row.update(status='error', error=repr(error))
            report['rows'].append(row)
            save()
            print(json.dumps(dict(dimension=int(dimension), solver=solver, status=row['status'])), flush=True)
            if row['status'] != 'ok':
                report['scale_up_stopped'] = True
                save()
                return
    report.update(status='ok', peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_method='process high-water RSS including compilation; separate process when --case-index and --solver are supplied')
    save()


if __name__ == '__main__':
    main()
