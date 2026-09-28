"""An external Flax XC network, trained in all three existing modes.

H2/6-31G*, native CPU integrals, float64. The initial HF orbitals and grid
come from GradSCF; every learned SCF uses the external energy functional.
The network adds a correction to Dirac exchange; correlation is omitted.
Teacher-generated targets demonstrate AD/training, not chemical accuracy.
Run: PYTHONPATH=src JAX_PLATFORMS=cpu python examples/neural_xc/external_functional.py
"""
from dataclasses import replace
from time import perf_counter

import jax
import jax.numpy as jnp
from flax import linen as nn

jax.config.update('jax_enable_x64', True)

from gradscf.scf import restricted_molecule_from_spec_with_jax_rks
from gradscf.model.neural_xc import ExternalFunctional
from gradscf.model.training import (
    MolecularTrainingConfig, MolecularTrainingDatum, NeuralXCTrainer,
    make_self_consistent_predictor, molecular_loss,
)


# Both the feature schema and the architecture belong to this example.
def make_inputs(state):
    return {'x': jnp.log1p(state.rho[:, None]),
            'rho': state.rho, 'weights': state.weights}


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(4, dtype=jnp.float64, param_dtype=jnp.float64)(x))
        return nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64)(x)[:, 0]


model = Network()


def energy(params, inputs):
    rho = inputs['rho']
    baseline = -.75*(3/jnp.pi)**(1/3)*jnp.maximum(rho, 1e-18)**(4/3)
    correction = .02*rho*model.apply(params, inputs['x'])
    return jnp.sum(inputs['weights']*(baseline+correction))


functional = ExternalFunctional(make_inputs, energy,
    init_fn=lambda key, x: model.init(key, x['x']))
reference = restricted_molecule_from_spec_with_jax_rks(
    atom='H 0 0 0; H 0 0 .74', basis='6-31g*', xc_spec='hf',
    grids_level=0, integral_backend='native')
params = functional.init_from_molecule(jax.random.PRNGKey(0), reference)
teacher = jax.tree.map(lambda x: x+.03, params)
config = MolecularTrainingConfig(mode='self_consistent', scf_gradient_mode='implicit',
    e0_total_mse_weight=1., scf_max_cycle=80, scf_damping=.2,
    scf_conv_tol_energy=1e-11, scf_conv_tol_density=1e-9,
    scf_eigenvalue_jitter=0., scf_implicit_diff_tolerance=1e-9,
    scf_implicit_diff_max_iter=40)
predict = make_self_consistent_predictor(functional, training_config=config)
target_energy, _ = predict(teacher, reference)
target_energy = jax.lax.stop_gradient(target_energy)
datum = MolecularTrainingDatum(reference, target_e0_total_h=target_energy)
print('Self-consistent teacher energy / Ha:', float(target_energy))

for mode, backward in (('fixed_density', 'implicit'),
                       ('self_consistent', 'unrolled'),
                       ('self_consistent', 'implicit')):
    started = perf_counter()
    cfg = replace(config, mode=mode, scf_gradient_mode=backward)
    before = float(molecular_loss(params, functional, datum, training_config=cfg)[0])
    trained = NeuralXCTrainer(functional, [datum]).kernel(
        steps=3, params=params, learning_rate=.002, training_config=cfg)
    after = float(molecular_loss(trained.params, functional, datum, training_config=cfg)[0])
    print(mode, backward, 'loss:', before, '->', after,
          'seconds:', round(perf_counter()-started, 2))
    if not jnp.isfinite(after) or after >= before:
        raise RuntimeError('Expected a finite loss reduction in this demonstration')

# Measured CPU float64 output (Apple M4 Pro, JAX 0.8.1):
# Self-consistent teacher energy / Ha: -1.0400905116545243
# fixed_density         4.1417802781e-06 -> 9.2387785597e-07
# self_consistent/unrolled 5.6129071029e-06 -> 1.6771517640e-06
# self_consistent/implicit 5.6129071029e-06 -> 1.6771517640e-06
# Three Adam updates per mode; seed 0. Targets are synthetic.
