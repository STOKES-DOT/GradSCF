"""A four-layer MLP: AO density matrix -> total XC energy.

H2/6-31G*, CPU float64. Four Dense layers mean three hidden layers and one
scalar output layer. The network predicts the entire XC energy in Hartree;
there is no additional XC baseline or grid integration of its output.

This fixed-basis example has synthetic teacher targets. A flattened AO matrix
is basis/order dependent and the MLP imposes no size consistency or exact XC
constraints. It demonstrates the external-functional interface, not accuracy.

Run: PYTHONPATH=src JAX_PLATFORMS=cpu python examples/neural_xc/density_matrix_mlp.py
"""
import jax
import jax.numpy as jnp
from flax import linen as nn

jax.config.update('jax_enable_x64', True)

from gradscf.scf import restricted_molecule_from_spec_with_jax_rks
from gradscf.model.neural_xc import ExternalFunctional
from gradscf.model.training import (
    MolecularTrainingConfig, MolecularTrainingDatum, NeuralXCTrainer,
    density_on_grid, make_self_consistent_predictor, molecular_loss,
)


# The input is the CURRENT spin-summed AO density matrix, supplied by GradSCF.
def make_inputs(state):
    return state.total_density_matrix


class XCNetwork(nn.Module):
    @nn.compact
    def __call__(self, density_matrix):
        x = density_matrix.reshape(-1)
        for width in (16, 16, 16):
            x = nn.tanh(nn.Dense(width, dtype=jnp.float64, param_dtype=jnp.float64)(x))
        x = nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        return .05 * x[0]  # Scalar XC energy in Ha; a small initial energy scale.


model = XCNetwork()
functional = ExternalFunctional(make_inputs, model.apply, init_fn=model.init)

# GradSCF supplies integrals and an HF initial density. Learned SCF then uses
# only the MLP for XC; the HF exchange energy is not retained as a baseline.
reference = restricted_molecule_from_spec_with_jax_rks(
    atom='H 0 0 0; H 0 0 .74', basis='6-31g*', xc_spec='hf',
    grids_level=0, integral_backend='native')
params = functional.init_from_molecule(jax.random.PRNGKey(0), reference)
dm = reference.rdm1.sum(axis=0)
print('Density-matrix shape:', dm.shape)
print('Initial XC energy / Ha:', float(functional.energy_from_molecule(params, reference)))
print('XC potential shape:', functional.potential(params, reference, dm).shape)
delta_dm = jnp.eye(dm.shape[0])
response = functional.kernel_action(params, reference, dm, delta_dm)
print('XC kernel action norm:', float(jnp.linalg.norm(response)))

# Both energy and density targets come from another parameter set, solved
# self-consistently. Replace them with reference labels for real training.
config = MolecularTrainingConfig(
    mode='self_consistent', scf_gradient_mode='implicit',
    e0_total_mse_weight=1., grid_density_mse_weight=1.,
    scf_max_cycle=80, scf_damping=.2,
    scf_conv_tol_energy=1e-11, scf_conv_tol_density=1e-9,
    scf_eigenvalue_jitter=0., scf_implicit_diff_tolerance=1e-9,
    scf_implicit_diff_max_iter=40)
predict = make_self_consistent_predictor(functional, training_config=config)
teacher = jax.tree.map(lambda x: x+.02, params)
target_energy, target_state = predict(teacher, reference)
datum = MolecularTrainingDatum(
    reference, target_e0_total_h=jax.lax.stop_gradient(target_energy),
    target_grid_density=jax.lax.stop_gradient(density_on_grid(target_state)))

before = float(molecular_loss(params, functional, datum, training_config=config)[0])
trained = NeuralXCTrainer(functional, [datum]).kernel(
    steps=5, params=params, learning_rate=.002, training_config=config)
after = float(molecular_loss(trained.params, functional, datum, training_config=config)[0])
print('Training loss:', before, '->', after)
print('Accepted updates:', trained.history['update_accepted'])
print('SCF converged:', trained.history['scf_converged'])
if not all(trained.history['update_accepted']) or not jnp.isfinite(after) or after >= before:
    raise RuntimeError('Expected five valid updates and a finite loss reduction')

# Measured output: CPU float64, JAX 0.8.1, random seed 0.
# Density-matrix shape: (4, 4)
# Initial XC energy / Ha: -0.0078110503267572184
# XC potential shape: (4, 4)
# XC kernel action norm: 0.01573187494899862
# Training loss: 2.15340029562302e-06 -> 1.8271876314836187e-06
# Accepted updates: [1.0, 1.0, 1.0, 1.0, 1.0]
# SCF converged: [1.0, 1.0, 1.0, 1.0, 1.0]
