"""Complete molecular geometry value/gradient after preparing the same seed HF/KS.

Each strategy runs in its own process. The seed reference construction is
reported separately. Timed calls rebuild moving integrals and grid and relax
the physical state; no PySCF data or fixed-density partial derivative is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward', choices=('minimize', 'scf'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import jax
    jax.config.update('jax_enable_x64', True)
    import numpy as np
    import gradscf
    from gradscf import dft, gto
    from gradscf.integrals._native import register_integrals
    from gradscf_tools.molecular_response import MolecularEnergy

    start = time.perf_counter()
    mf = dft.RKS(gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
        basis='def2-svp', cart=True), xc='pbe', grids_level=2,
        conv_tol=1e-12, conv_tol_grad=1e-10, conv_tol_density=1e-10).run()
    assert mf.converged
    seed_seconds = time.perf_counter()-start
    energy = MolecularEnergy(mf, forward=args.forward, gradient_tolerance=1e-9)
    value = jax.jit(energy)
    derivative = jax.jit(jax.value_and_grad(energy))
    report = dict(forward=args.forward, python=sys.executable, source=gradscf.__file__,
        config=dict(atom=mf.mol.atom, basis='def2-svp', cart=True, xc='pbe', grid_level=2,
                    energy_unit='Hartree', coordinate_unit='Bohr', gradient_unit='Hartree/Bohr',
                    dtype='float64', threads=1, gradient_tolerance=1e-9),
        timing_scope='full moving geometry workflow after fixed reference/seed preparation',
        seed_reference_seconds=seed_seconds, seed_scf_cycles=int(mf.cycles), samples=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    for kind, function in (('value', value), ('value_and_grad', derivative)):
        for repeat in range(4):
            tick = time.perf_counter()
            result = jax.block_until_ready(function(energy.coordinates))
            elapsed = time.perf_counter()-tick
            e, gradient = (result, None) if kind == 'value' else result
            assert np.isfinite(e)
            if gradient is not None:
                assert np.isfinite(gradient).all()
            report['samples'].append(dict(kind=kind, repeat=repeat,
                cache='first' if repeat == 0 else 'compiled fixed-geometry execution',
                seconds=elapsed, energy=float(e), gradient=None if gradient is None else np.asarray(gradient).tolist()))
            save()
    moved = energy.coordinates.at[1, 2].add(.01)
    tick = time.perf_counter()
    e, gradient = jax.block_until_ready(derivative(moved))
    report['new_geometry'] = dict(seconds=time.perf_counter()-tick, energy=float(e),
                                  gradient=np.asarray(gradient).tolist())
    # Extra state certification is outside all clean timings.
    state = jax.block_until_ready(jax.jit(energy.solve)(energy.coordinates))
    report['state'] = dict(stationary=bool(state.stationary),
        gradient_norm=float(state.gradient_norm), iterations=int(state.iterations),
        evaluations=int(state.evaluations), density=np.asarray(state.density_matrix).tolist())
    assert state.stationary
    library = Path(register_integrals()._name)
    report.update(status='ok', peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_method='Linux ru_maxrss including compilation and prior repeats',
        native=str(library), native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        affinity=sorted(os.sched_getaffinity(0)))
    save()
    print(json.dumps(dict(status=report['status'], forward=args.forward,
                         samples=[s['seconds'] for s in report['samples']])))


if __name__ == '__main__':
    main()
