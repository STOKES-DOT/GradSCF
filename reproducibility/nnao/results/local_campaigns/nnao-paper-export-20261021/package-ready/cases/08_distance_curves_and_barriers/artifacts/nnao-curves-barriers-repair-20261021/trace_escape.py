"""Diagnose real orbital curvature and the energy of actual restart steps."""
from pathlib import Path
import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'tests/comparisons')
from nnao_core.native import _solver, make_inputs
from nnao_reactivity.rescue import solve_native, orbital_stability, _physical
from gradscf.scf.orbital_optimization import _rotation_functions
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.integrals.molecular.jk import build_jk_from_df


def _relax_orbitals(coeff, occupations, hcore, factors, enuc, spin):
    # Diagnostic only: production rescue uses negative-mode rotations and DIIS.
    from gradscf.solvers.nonlinear.minimize import solve_orbitals
    dimension, rotate, density_from = _rotation_functions('uks', occupations)
    def energy(angles, base, physical):
        h, b, nuc = physical
        ds = density_from(rotate(angles, base))
        total = ds.sum(axis=0)
        coulomb = build_jk_from_df(b, total)[0]
        exchange = sum(jnp.sum(d*build_jk_from_df(b, d)[1]) for d in ds)
        return jnp.sum(total*h)+.5*jnp.sum(total*coulomb)-.5*exchange+nuc
    result = solve_orbitals((hcore, factors, enuc), coeff, energy=energy, rotate=rotate,
        dimension=dimension, max_iterations=100, gradient_tolerance=1e-9)
    return density_from(result.coeff), dict(orbital_iterations=int(result.iterations),
        evaluations=int(result.evaluations), orbital_gradient_norm=float(result.gradient_norm),
        energy=float(energy(jnp.zeros(dimension), result.coeff, (hcore, factors, enuc))))

root = Path(__file__).resolve().parent
parent = Path('artifacts/nnao-curves-barriers-20261020/representative')
records = json.loads((parent/'results.json').read_text())['results']
old = next(r for r in records if r['identifier'] == 'BH9_06_1R2' and r['basis'] == 'sto-6g')
export = json.loads((parent/f'{old["key"]}-basis.json').read_text())
row, shells = export['row'], export['atom_shells']
data = make_inputs(row, shells)
inputs = tuple(map(jnp.asarray, (data['overlap'], data['hcore'], data['factors'], data['enuc'])))
state = solve_native(*inputs[:3], row['nelectron'], row['spin'], inputs[-1],
                     row=row, atom_shells=shells, check_stability=False)
b = unpack_factors(inputs[2], len(inputs[0]))
stability = orbital_stability(state, inputs[1], b, row['spin'])
report = dict(identifier=old['identifier'], basis=old['basis'], energy=float(state['energy']),
              stability={k:v for k,v in stability.items() if k != 'direction'}, trials=[])
print(report, flush=True)
occ = tuple(map(tuple, np.asarray(state['occupations'])))
_, rotate, density_from = _rotation_functions('uks', occ)
for step in [.1, -.1, .5, -.5, 1., -1.]:
    coeff = rotate(step*stability['direction'], jnp.asarray(state['mo']))
    density = density_from(coeff)
    _, physical = _physical(dict(density_spin=density, converged=False), inputs[0], inputs[1], b,
                             inputs[-1], row['nelectron'], row['spin'])
    candidate = _solver(row['nelectron'], row['spin'], 0.)(*inputs, density)
    result = dict(step=step, trial_energy=float(physical['energy']),
        trial_change_hartree=float(physical['energy'])-float(state['energy']),
        scf_energy=float(candidate['energy']), scf_converged=bool(candidate['converged']),
        scf_density_change=float(jnp.linalg.norm(candidate['density_spin']-state['density_spin'])))
    report['trials'].append(result)
    print(result, flush=True)
    (root/'trace-escape.json').write_text(json.dumps(report, indent=2)+'\n')

rotated = rotate(stability['direction'], jnp.asarray(state['mo']))
density, info = _relax_orbitals(rotated, occ, inputs[1], b, inputs[-1], row['spin'])
valid, physical = _physical(dict(density_spin=density, converged=True), inputs[0], inputs[1], b,
                           inputs[-1], row['nelectron'], row['spin'])
candidate = _solver(row['nelectron'], row['spin'], 0.)(*inputs, density)
report['orbital_relaxation'] = dict(**info, raw_energy=float(physical['energy']),
    raw_orbital_residual=physical['orbital_residual'],
    raw_idempotency=physical['idempotency_residual'],
    polished_energy=float(candidate['energy']), polished_converged=bool(candidate['converged']))
print(report['orbital_relaxation'], flush=True)
(root/'trace-escape.json').write_text(json.dumps(report, indent=2)+'\n')
