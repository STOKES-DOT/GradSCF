"""Optimize gas-phase anthracene with fresh SCF states and relaxed AD forces.

The numerical SCF orbitals are dynamic inputs to the reference-mode implicit
response. No old density or orbital solution is accepted at a new geometry.
All coordinates are bohr and gradients are Hartree/bohr. No CLI.
"""
import hashlib
import json
from time import perf_counter
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from common import OUTPUT, SETTINGS, event, mean_field, save
from gradscf_tools.molecular_response import MolecularEnergy


def checkpoint(name, **arrays):
    target = OUTPUT / name
    temporary = target.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(target)


def main():
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    signature = hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()
    save('geometry_status.json', dict(complete=False, converged=False, stage='starting',
         settings_sha256=signature))
    restart = OUTPUT/'optimization_latest.npz'
    source = restart if restart.exists() else OUTPUT/'initial_scf.npz'
    with np.load(source) as data:
        if 'settings_sha256' in data and str(data['settings_sha256']) != signature:
            raise RuntimeError('The optimization checkpoint uses different settings.')
        coordinates, density = data['coordinates'].copy(), data['density'].copy()
    seed = mean_field(coordinates, density)
    energy = MolecularEnergy(seed, forward='reference', jk_backend='direct',
        gradient_tolerance=max(1e-8, 10*SETTINGS['scf_gradient_tol']),
        max_memory=SETTINGS['max_memory_mb'])
    calculate = jax.jit(jax.value_and_grad(
        lambda r, coefficients: energy(r, solution=coefficients), argnums=0))
    history, latest = [], {}
    started = perf_counter()

    def objective(flat):
        nonlocal density
        r = flat.reshape(coordinates.shape)
        checkpoint('optimization_trial.npz', coordinates=r, density=density,
                   settings_sha256=signature)
        begin = perf_counter()
        # The just-converged setup state is already the first trial's SCF.
        mf = seed if not history and np.array_equal(r, coordinates) else mean_field(r, density)
        value, gradient = calculate(jnp.asarray(r), jnp.asarray(mf.mo_coeff))
        value, gradient = float(value), np.asarray(gradient)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError('The converged electronic state failed its implicit force check.')
        if abs(value-float(mf.e_tot)) > 1e-8:
            raise RuntimeError('SCF and implicit-reference energies do not agree.')
        translation = float(np.max(np.abs(gradient.sum(axis=0))))
        if translation > 2e-6:
            raise RuntimeError(f'Translation-gradient sum rule failed: {translation:.3e}')
        if not history:
            direction = np.zeros_like(r)
            direction[0, 0], direction[1, 0] = 1/np.sqrt(2), -1/np.sqrt(2)
            step = 1e-4
            values = [float(mean_field(r+sign*step*direction, mf.make_rdm1()).e_tot)
                      for sign in (1, -1)]
            finite_difference = (values[0]-values[1])/(2*step)
            analytic = float(np.sum(gradient*direction))
            error = abs(analytic-finite_difference)
            save('ground_force_check.json',dict(step_bohr=step,
                analytic_ha_per_bohr=analytic,fd_ha_per_bohr=finite_difference,
                absolute_error=error,tolerance=5e-6,complete=error<=5e-6))
            if error > 5e-6:
                raise RuntimeError('Initial complete geometric force failed its FD gate.')
        density = np.asarray(mf.make_rdm1())
        latest.update(coordinates=r.copy(), energy=value, gradient=gradient,
            mo_coeff=np.asarray(mf.mo_coeff), mo_energy=np.asarray(mf.mo_energy),
            mo_occ=np.asarray(mf.mo_occ), density=density, settings_sha256=signature)
        checkpoint('optimization_latest.npz', **latest)
        record = dict(evaluation=len(history)+1, energy_ha=value,
            max_gradient_ha_per_bohr=float(np.max(np.abs(gradient))),
            translation_gradient=translation, scf_cycles=int(mf.cycles),
            seconds=perf_counter()-begin)
        history.append(record)
        save('geometry_status.json', dict(complete=False, history=history,
             settings_sha256=signature, elapsed_seconds=perf_counter()-started))
        event('geometry_evaluation', **record)
        return value, gradient.ravel()

    minimum = minimize(objective, coordinates.ravel(), jac=True, method='BFGS',
        options=dict(gtol=SETTINGS['geometry_gradient_tol'],
                     maxiter=SETTINGS.get('geometry_maxiter', 120)))
    # The last line-search evaluation need not be the returned minimum.
    if not np.array_equal(latest['coordinates'].ravel(), minimum.x):
        objective(minimum.x)
    maximum = float(np.max(np.abs(latest['gradient'])))
    accepted = maximum <= SETTINGS['geometry_gradient_tol']
    if accepted:
        checkpoint('ground.npz', **latest)
    save('geometry_status.json', dict(complete=accepted, converged=accepted,
        optimizer_success=bool(minimum.success), optimizer_message=str(minimum.message),
        iterations=int(minimum.nit), history=history, settings_sha256=signature,
        max_gradient_ha_per_bohr=maximum, elapsed_seconds=perf_counter()-started,
        coordinate_units='bohr', gradient_units='Hartree/bohr'))
    if not accepted:
        raise RuntimeError(f'Geometry gradient {maximum:.3e} exceeds the requested tolerance.')
    event('geometry_complete', energy_ha=latest['energy'], max_gradient_ha_per_bohr=maximum)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('geometry_failure.json', dict(complete=False, error=traceback.format_exc(),
             checkpoint='optimization_latest.npz', failed_trial='optimization_trial.npz'))
        raise
