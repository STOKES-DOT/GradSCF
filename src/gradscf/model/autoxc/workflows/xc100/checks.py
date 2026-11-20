"""Independent molecular PBE acceptance checks; PySCF is an optional oracle only."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from .pbe import energy_density


def baseline_checks(case):
    """Check the archive, weak-form PBE matrix and scalar-energy AD independently."""
    from pyscf import gto
    from pyscf.dft import numint
    started = time.perf_counter()
    a = case.arrays
    mol = gto.M(atom=case.atom_string, basis=case.basis_definitions, unit='Bohr',
                charge=case.metadata['charge'], spin=0, cart=True, verbose=0)
    integrator = numint.NumInt()
    t = np.asarray(case.transform)
    metric = t.T@mol.intor_symmetric('int1e_ovlp')@t
    result = {'overlap_max_error': float(np.max(np.abs(metric-a['overlap'])))}
    matrix = np.zeros((mol.nao, mol.nao))
    energy, field_error, gradient_error = 0., 0., 0.
    for start in range(0, case.grid.ngrids, 4096):
        stop = min(start+4096, case.grid.ngrids)
        ao = numint.eval_ao(mol, a['grid_coords_bohr'][start:stop], deriv=1)
        weights = a['grid_weights'][start:stop]
        n = np.einsum('gi,ij,gj->g', ao[0], a['density_cartesian'], ao[0], optimize=True)
        grad = 2*np.einsum('xgi,ij,gj->xg', ao[1:], a['density_cartesian'], ao[0], optimize=True)
        field_error = max(field_error, float(np.max(np.abs(n-a['rho'][start:stop]))))
        gradient_error = max(gradient_error, float(np.max(np.abs(grad.T-a['grad_rho'][start:stop]))))
        effective = np.sqrt(n*n+case.floor**2)
        exc, vxc, _, _ = integrator.eval_xc('PBE', np.vstack((effective, grad)), spin=0, deriv=1)
        energy += np.dot(weights, effective*exc)
        matrix += np.einsum('g,gi,gj->ij', weights*vxc[0]*n/effective, ao[0], ao[0], optimize=True)
        gradient_term = np.einsum('xg,xgi,gj->ij', 2*weights*vxc[1]*grad,
                                 ao[1:], ao[0], optimize=True)
        matrix += gradient_term+gradient_term.T
    result.update(oracle_xc_energy_hartree=float(energy), xc_energy_hartree=case.baseline_energy,
        energy_error_hartree=abs(case.baseline_energy-energy),
        potential_max_error_hartree=float(np.max(np.abs(np.asarray(case.vxc)-t.T@matrix@t))),
        density_field_max_error=field_error, density_gradient_max_error=gradient_error)

    direction = case.target.density/jnp.linalg.norm(case.target.density)
    energy_fn = lambda d: case._energy(case.baseline_functional, None, d)
    analytic = float(jnp.einsum('ij,ji->', case.vxc, direction))
    fd = []
    for step in (1e-3, 3e-4, 1e-4):
        numerical = float((energy_fn(case.target.density+step*direction)-
                           energy_fn(case.target.density-step*direction))/(2*step))
        fd.append({'step': step, 'derivative': numerical, 'absolute_error': abs(numerical-analytic)})
    score = case.reporter.score(case.vxc)
    shifted = case.problem.score(case.vxc+.3*case.target.overlap)
    result.update(energy_directional_derivative=analytic, finite_differences=fd,
        W_hartree=float(score.value), variational_gap_hartree=float(score.variational_gap),
        gap_hartree=float(score.gap), relative_density_l2=float(score.density_error),
        baseline_reward=float(score.reward), gauge_error_hartree=abs(float(shifted.value-score.value)),
        grid_electron_error=abs(float(np.dot(a['grid_weights'], a['rho']))-case.target.nelec),
        seconds=time.perf_counter()-started)
    gates = {'overlap': result['overlap_max_error'] < 1e-10,
        'fields': field_error < 1e-9 and gradient_error < 1e-8,
        'energy': result['energy_error_hartree'] < 1e-8,
        'potential': result['potential_max_error_hartree'] < 1e-7,
        'energy_ad': fd[-1]['absolute_error'] < 1e-6,
        'spectrum': bool(score.valid), 'baseline_zero': score.reward == 0.,
        'gauge': result['gauge_error_hartree'] < 1e-8,
        'quadrature': result['grid_electron_error'] < 1e-6}
    result['gates'] = {k: bool(v) for k, v in gates.items()}
    result['passed'] = all(result['gates'].values())
    return result


def candidate_checks(bound, params):
    """AD, density-moment identity and FD refer to one unchanged affine candidate."""
    names = list(params)
    point = jnp.asarray([float(params[n])+.017 for n in names])
    unpack = lambda x: dict(zip(names, x))
    objective = jax.jit(lambda x: bound(unpack(x)).value)
    value, gradient = jax.value_and_grad(objective)(point)
    score = bound(unpack(point))
    expected = jnp.asarray([jnp.einsum('ij,ji->', bound.directions[n],
                            score.density-bound.case.target.density) for n in names])
    numerical = []
    for index in range(len(names)):
        direction = jnp.eye(len(names))[index]*1e-4
        numerical.append(float((objective(point+direction)-objective(point-direction))/(2e-4)))
    fd_error = float(jnp.max(jnp.abs(jnp.asarray(numerical)-gradient))) if names else 0.
    identity_error = float(jnp.max(jnp.abs(expected-gradient))) if names else 0.
    return dict(value_hartree=float(value), ad_gradient=np.asarray(gradient).tolist(),
                identity_gradient=np.asarray(expected).tolist(), finite_difference=numerical,
                fd_max_error=fd_error, identity_max_error=identity_error,
                passed=bool(fd_error < 1e-6 and identity_error < 1e-9 and score.valid))
