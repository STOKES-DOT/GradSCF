"""C/STO-3G, grid level 1: FCI -> auxiliary inverse reference -> XC assessment.

The three degenerate 3P roots at Ms=1 are equally averaged. A finite-temperature
matrix inverse is projected into a local AO-product potential basis. Its
temperature is an auxiliary regularizer, not a change to the zero-temperature
FCI energy labels. This is fixed-reference pretraining, not candidate SCF.
Run from the repository root: PYTHONPATH=src JAX_PLATFORMS=cpu python examples/iks/carbon_fci.py
"""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import csv
import json
import platform
import os
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update('jax_enable_x64', True)

from gradscf import gto, scf, fci, dft, integrals
from gradscf.dft import iks
from gradscf.integrals.grids.blocks import AOGrid
from gradscf.model import neural_xc
from gradscf.model.neural_xc import training


folder = Path('artifacts/iks-carbon-sto3g-20261026')
folder.mkdir(parents=True, exist_ok=True)
started = perf_counter()
mol = gto.M(atom='C 0 0 0', basis='sto-3g', spin=2, charge=0, cart=True)
common = scf.ROHF(mol, grids_level=1, execution_device='cpu',
    conv_tol=1e-12, conv_tol_grad=1e-10, max_cycle=100).run()
solver = fci.FCI(common, solver='dense', nroots=3, conv_tol=1e-11).run()
if not common.converged or not np.all(solver.converged):
    raise RuntimeError('Converge the mean-field and FCI references')
np.testing.assert_allclose(solver.e_tot, solver.e_tot[0], atol=1e-10, rtol=0)
coefficient = common.mo_coeff
root_densities = []
for root in range(3):
    np.testing.assert_allclose(solver.spin_square(root=root)[0], 2., atol=1e-10)
    root_densities.append(coefficient@jnp.stack(solver.make_rdm1s(root=root))@coefficient.T)
target_density = jnp.mean(jnp.stack(root_densities), axis=0)
target_energy = jnp.mean(solver.e_tot)

# The UKS-shaped reference supplies the AO grid and existing model inputs.
# ROHF.to_reference() is not used: its response adapter is not implemented.
source = scf.UHF(mol, grids_level=1, execution_device='cpu',
    conv_tol=1e-12, conv_tol_grad=1e-10, max_cycle=100).run()
if not source.converged: raise RuntimeError('UHF input preparation failed')
state = source.to_reference()
target = iks.Target(target_density, state.overlap_matrix, (4, 2), layout='uks', energy=target_energy)
ao_basis = integrals.basis_from_molecule_spec(mol.to_spec(), basis='sto-3g')
grid = AOGrid.from_basis(ao_basis, state.grid.coords, state.grid.weights, block_size=512)
potential = iks.Potential.ao_products(grid, state.overlap_matrix, layout='uks', max_memory=1024.)
temperature = .01  # Hartree, auxiliary Fermi regularization (kB=1).
inverse = iks.invert(target, potential, jnp.stack((state.h1e, state.h1e)),
    method='thermal_matrix', temperature=temperature, regularization=1e-12, tolerance=1e-6)
target_rho = training.density_on_grid(replace(state, rdm1=target_density))
inverse_rho = training.density_on_grid(replace(state, rdm1=inverse.density))
relative_density_error = jnp.sqrt(jnp.sum(state.grid.weights*(inverse_rho-target_rho)**2)
                                /jnp.sum(state.grid.weights*target_rho**2))
report = dict(atom='C', charge=0, spin=2, basis='sto-3g', cart=True, grid_level=1,
    grid_points=int(state.grid.weights.size), fci_determinants=solver.space.size,
    fci_roots=3, target_density='equal mixture of three degenerate 3P roots at Ms=1',
    fci_total_energy_hartree=float(target_energy), fci_root_energies_hartree=np.asarray(solver.e_tot).tolist(),
    inverse_method=inverse.method, auxiliary_temperature_hartree=temperature,
    inverse_density_error=float(inverse.density_error), inverse_operator_error_hartree=float(inverse.operator_error),
    inverse_linear_residual=float(inverse.linear_residual), inverse_valid=bool(inverse.valid),
    inverse_projection_rank=int(inverse.projection_rank),
    inverse_projection_condition=float(inverse.projection_condition),
    inverse_regularization=1e-12, inverse_rank_tolerance=1e-12, inverse_tolerance=1e-6,
    inverse_grid_density_relative_l2=float(relative_density_error),
    backend=jax.default_backend(), jax=jax.__version__, python=platform.python_version(),
    platform=platform.platform(), machine=platform.machine(), devices=list(map(str, jax.devices())),
    thread_environment={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},
    dtype='float64', runs={})
(folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print('FCI total energy / Ha:', float(target_energy), flush=True)
print('Inverse validity/density/operator errors:', bool(inverse.valid),
      float(inverse.density_error), float(inverse.operator_error), flush=True)
if not inverse.valid:
    raise RuntimeError('The declared inverse tolerances were not met; no teacher training performed')
np.savez_compressed(folder/'reference.npz', density_ao=np.asarray(target_density),
    root_densities_ao=np.asarray(root_densities), overlap=np.asarray(target.overlap),
    recovered_density_ao=np.asarray(inverse.density), recovered_fock=np.asarray(inverse.fock),
    potential_coefficients=np.asarray(inverse.potential_coefficients),
    grid_coordinates=np.asarray(state.grid.coords), grid_weights=np.asarray(state.grid.weights),
    target_density_grid=np.asarray(target_rho), recovered_density_grid=np.asarray(inverse_rho),
    energy_hartree=np.asarray(target_energy))


def dirac_channels(features):
    c = .75*(6/jnp.pi)**(1/3)
    return -c*(jnp.maximum(features.rho_a, 1e-18)**(4/3)
               +jnp.maximum(features.rho_b, 1e-18)**(4/3))


analytic = dft.custom_xc(
    input_fn=lambda fields: dict(rho=fields.rho_spin, weights=fields.weights),
    energy_fn=lambda p, x: -.75*(6/jnp.pi)**(1/3)*jnp.sum(
        x['weights']*jnp.sum(jnp.maximum(x['rho'], 1e-18)**(4/3), axis=0)),
    name='spin_Dirac_exchange')
analytic_energy = iks.energy_function(state, analytic, layout='uks', nuclear_repulsion=0.)
analytic_result = iks.evaluate(inverse, analytic_energy, {})
report['analytic'] = dict(total_energy_hartree=float(analytic_result.total_energy),
    energy_error_hartree=float(analytic_result.energy_error), operator_mse_hartree2=float(analytic_result.operator_mse),
    stationarity_mse=float(analytic_result.stationarity_mse), valid=bool(analytic_result.valid))

# Reuse the existing residual mixing network; only its supplied energy channel
# is handwritten, so this example does not require installing jax-xc locally.
model = neural_xc.Functional(semilocal_energy_density_fn=dirac_channels,
    n_semilocal_channels=1, hidden_dims=(16, 16), include_hfx_channel=False,
    ground_state_hf_mode='off', ground_state_pt2_mode='off', name='carbon_residual_xc')
initial_state = replace(state, rdm1=inverse.density)
params = model.init_from_molecule(jax.random.PRNGKey(0), initial_state)
params = jax.tree.map(lambda p: p.astype(jnp.float64), params)
energy = iks.energy_function(state, model, layout='uks', nuclear_repulsion=0.)
operator_weight, steps, learning_rate = .01, 100, .002


def loss(parameters):
    result = iks.evaluate(inverse, energy, parameters)
    error = result.energy_error
    value = error**2+jnp.abs(error)+operator_weight*result.operator_mse
    return jnp.where(result.valid, value, jnp.nan), result


optimizer = optax.adam(learning_rate)
optimizer_state = optimizer.init(params)


@jax.jit
def step(parameters, opt_state):
    (_, _), gradient = jax.value_and_grad(loss, has_aux=True)(parameters)
    updates, opt_state = optimizer.update(gradient, opt_state, parameters)
    return optax.apply_updates(parameters, updates), opt_state


evaluate = jax.jit(loss)
rows = []
for iteration in range(steps+1):
    value, result = evaluate(params)
    if not result.valid or not np.isfinite(float(value)):
        raise RuntimeError('Invalid functional evaluation; training stopped')
    rows.append(dict(step=iteration, loss=float(value), total_energy_hartree=float(result.total_energy),
        energy_error_hartree=float(result.energy_error), operator_mse_hartree2=float(result.operator_mse),
        stationarity_mse=float(result.stationarity_mse)))
    if iteration % 20 == 0: print('step', iteration, rows[-1], flush=True)
    if iteration < steps: params, optimizer_state = step(params, optimizer_state)

direction = jax.tree.map(lambda p: .001*jnp.sin(jnp.arange(p.size).reshape(p.shape)+.3), params)
directional_ad = jax.jit(lambda p: jax.jvp(lambda q: loss(q)[0], (p,), (direction,))[1])(params)
checks = []
for displacement in (1e-3, 3e-4):
    plus = jax.tree.map(lambda p, v: p+displacement*v, params, direction)
    minus = jax.tree.map(lambda p, v: p-displacement*v, params, direction)
    finite = (evaluate(plus)[0]-evaluate(minus)[0])/(2*displacement)
    np.testing.assert_allclose(directional_ad, finite, atol=2e-7, rtol=2e-5)
    checks.append(dict(step=displacement, automatic=float(directional_ad), finite_difference=float(finite),
                       absolute_error=float(jnp.abs(directional_ad-finite))))

with (folder/'training.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
from flax.serialization import to_bytes
(folder/'params.msgpack').write_bytes(to_bytes(params))
report['runs']['fixed_inverse_reference'] = dict(initial=rows[0], final=rows[-1],
    steps=steps, seed=0, learning_rate=learning_rate, operator_weight=operator_weight,
    architecture='existing ResidualMixingMLP, hidden_dims=(16,16), one spin-Dirac energy channel',
    parameter_count=sum(p.size for p in jax.tree.leaves(params)),
    parameter_initialization='Flax default float32 initialization, explicitly lifted to float64 before training',
    optimizer='Adam', loss='delta_E**2 + abs(delta_E) + 0.01*operator_mse; numerical energies in Hartree',
    density_supervision='FCI density defines the inverse operator target; no candidate SCF density loss',
    energy_error_below_1mHa=abs(rows[-1]['energy_error_hartree']) < .001,
    minimum_loss_during_run=min(rows, key=lambda row: row['loss']),
    parameter_directional_checks=checks,
    total_seconds_including_reference_and_compile=perf_counter()-started)
report['example_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
iterations = [row['step'] for row in rows]
axes[0].semilogy(iterations, [row['loss'] for row in rows], color='#245b8c')
axes[0].set_ylabel('Training loss')
axes[1].semilogy(iterations, [abs(row['energy_error_hartree'])*1000 for row in rows], color='#245b8c')
axes[1].axhline(abs(report['analytic']['energy_error_hartree'])*1000,
               color='#b36a30', linestyle='--', label='Spin Dirac exchange')
axes[1].set_ylabel('Absolute total-energy error (mHa)')
axes[1].legend(frameon=False, fontsize=8)
axes[2].plot(iterations, [np.sqrt(row['operator_mse_hartree2']) for row in rows], color='#245b8c', label='Neural XC')
axes[2].axhline(np.sqrt(report['analytic']['operator_mse_hartree2']),
               color='#b36a30', linestyle='--', label='Spin Dirac exchange')
axes[2].set_ylabel('Aligned operator RMSE (Ha)')
axes[2].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.set_xlabel('Adam step'); ax.grid(alpha=.2)
fig.suptitle('C / STO-3G, grid level 1 — fixed auxiliary inverse reference')
fig.savefig(folder/'training.png', dpi=180)
fig.savefig(folder/'training.pdf')
plt.close(fig)
print('Results:', folder, flush=True)

# Measured local CPU float64, seed 0, 100 Adam updates, auxiliary T=0.01 Ha:
# FCI total energy / Ha: -37.21873355063647
# Inverse metric-density max error: 2.285e-7; grid relative L2 error: 5.806e-9.
# Training loss: 7.602335 -> 0.053638.
# Last-step energy error / Ha: +0.024865 (not a converged 1 mHa fit).
# Operator MSE / Ha^2: 2.346363 -> 2.815461 (increased).
# Energy fitting and operator quality are separate; no candidate SCF was run.
