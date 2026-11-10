"""Analytic ground-state Hessian columns and all 66 anthracene normal modes.

One converged SCF reference supplies the stationary root. Every column is a
JVP of its full implicit AD gradient; the SCF loop is not rerun per column.
Checkpoints retain the exact geometry, source, settings and native identity.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from common import OUTPUT, ROOT, SETTINGS, SYMBOLS, event, mean_field, save
from gradscf.integrals._native import register_integrals
from gradscf_tools.molecular_response import MolecularEnergy
from gradscf_tools.vibrations import normal_modes, AMU_TO_ELECTRON_MASS


def main():
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    status = json.loads((OUTPUT/'geometry_status.json').read_text())
    if not status.get('complete') or not status.get('converged'):
        raise RuntimeError('An accepted ground-state minimum is required.')
    save('harmonics_status.json', dict(complete=False, stage='starting'))
    with np.load(OUTPUT/'ground.npz') as data:
        r, c = jnp.asarray(data['coordinates']), jnp.asarray(data['mo_coeff'])
        density, saved_energy = data['density'].copy(), float(data['energy'])
    mf = mean_field(np.asarray(r), density)
    if abs(float(mf.e_tot)-saved_energy) > 1e-8:
        raise RuntimeError('The harmonic reference differs from the accepted ground state.')
    energy = MolecularEnergy(mf, forward='reference', jk_backend='direct',
        gradient_tolerance=max(1e-8, 10*SETTINGS['scf_gradient_tol']),
        max_memory=SETTINGS['max_memory_mb'])
    scalar = lambda x, coefficients: energy(x, solution=coefficients)
    gradient = jax.jit(jax.grad(scalar, argnums=0))
    hvp = jax.jit(lambda x, coefficients, v:
                  jax.jvp(jax.grad(scalar, argnums=0), (x, coefficients),
                          (v, jnp.zeros_like(coefficients)))[1])
    force = np.asarray(gradient(r, c))
    if not np.isfinite(force).all() or np.max(np.abs(force)) > SETTINGS['geometry_gradient_tol']:
        raise RuntimeError('The harmonic reference is not a stationary geometry.')
    native = Path(register_integrals()._name).resolve()
    digest = hashlib.sha256()
    for path in sorted((ROOT/'src').rglob('*.py')):
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(path.read_bytes())
    identity = dict(source_sha256=digest.hexdigest(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        coordinates_sha256=hashlib.sha256(np.asarray(r).tobytes()).hexdigest(),
        coefficients_sha256=hashlib.sha256(np.asarray(c).tobytes()).hexdigest(),
        settings=SETTINGS, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = OUTPUT/'hessian_columns'
    directory.mkdir(exist_ok=True)
    columns, timings = [], []
    started = perf_counter()
    for index, direction in enumerate(np.eye(r.size).reshape(-1, *r.shape)):
        path = directory/f'column_{index:03d}.npz'
        if path.exists():
            with np.load(path) as data:
                if str(data['signature']) != signature:
                    raise RuntimeError(f'Hessian checkpoint identity mismatch: {path.name}')
                column, seconds = data['column'].copy(), float(data['seconds'])
        else:
            event('hessian_column', column=index+1, total=int(r.size))
            begin = perf_counter()
            column = np.asarray(hvp(r, c, jnp.asarray(direction))).ravel()
            seconds = perf_counter()-begin
            if not np.isfinite(column).all():
                raise RuntimeError(f'Hessian column {index+1} is nonfinite.')
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, signature=signature, column=column, seconds=seconds)
            temporary.replace(path)
        columns.append(column); timings.append(seconds)
        event('hessian_column_saved', column=index+1, seconds=seconds)
    hessian = np.column_stack(columns)
    translations = np.tile(np.eye(3), (len(r), 1))
    checks = dict(raw_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
        translation_residual=float(np.max(np.abs(hessian@translations))))
    if any(value > 2e-6 for value in checks.values()):
        raise RuntimeError(f'Raw Hessian checks failed: {checks}')
    masses = np.array([12. if symbol == 'C' else 1.00782503223 for symbol in SYMBOLS])
    modes = normal_modes(hessian, np.asarray(r), masses)
    if modes.rigid_rank != 6 or modes.displacements.shape[-1] != 66 or np.any(modes.frequencies_cm1 <= 0):
        raise RuntimeError('Expected 66 positive internal modes; no imaginary frequency is clipped.')
    np.savez(OUTPUT/'harmonics.npz', coordinates=np.asarray(r), energy=saved_energy,
        gradient=force, hessian=hessian, frequencies_cm1=modes.frequencies_cm1,
        modes_bohr_per_Q=modes.displacements, masses_amu=masses,
        amu_to_electron_mass=AMU_TO_ELECTRON_MASS, signature=signature)
    save('harmonics_status.json', dict(complete=True, **checks, identity=identity,
        signature=signature, rigid_rank=modes.rigid_rank, rigid_residual=modes.rigid_residual,
        frequencies_cm1=modes.frequencies_cm1.tolist(), column_seconds=timings,
        elapsed_seconds=perf_counter()-started,
        coordinate_units='bohr', mode_coordinate_units='bohr sqrt(electron mass)',
        hessian_units='Hartree/bohr^2', construction='JVP of the complete implicit AD gradient'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('harmonics_failure.json', dict(complete=False, error=traceback.format_exc()))
        raise
