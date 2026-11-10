"""Fixed-orbital XC assessment, parameter fitting and agent-style metrics.

A manufactured two-AO problem demonstrates the public interfaces without an
integral build or an inverse calculation. These supplied orbitals are not a
Wu-Yang result. The AO-basis quadratic and MLP below are interface examples,
not transferable physical XC approximations. Atomic units, CPU, float64.
"""
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from gradscf import dft
from gradscf.dft import iks
from gradscf.model.neural_xc import training
from gradscf.tools.features import MoleculeLikeState, MoleculeGridView


# 1. A supplied nonorthogonal orbital reference with a known target energy.
s = jnp.diag(jnp.array([1.2, .8]))
root = jnp.diag(jnp.sqrt(jnp.diag(s)))
c = jnp.stack((jnp.linalg.inv(root),)*2)
occupations = jnp.array([[1., 0.], [1., 0.]])
density = (c*occupations[:, None, :])@c.transpose(0, 2, 1)
target = iks.Target(density, s, (1, 1), layout='uks', energy=-.72)
reference = iks.OrbitalReference(target, c, occupations,
    jnp.array([[-.32, .6], [-.32, .6]]), method='manufactured_two_AO')
state = MoleculeLikeState(
    ao=jnp.array([[1., .2], [.3, .8]]), ao_deriv1=jnp.zeros((4, 2, 2)),
    grid=MoleculeGridView(weights=jnp.ones(2)), rdm1=density,
    mo_coeff=c, mo_occ=occupations, mo_energy=reference.energies,
    h1e=root@jnp.diag(jnp.array([-.4, .6]))@root,
    rep_tensor=jnp.zeros((2,)*4), overlap_matrix=s)
sample = training.Sample(state, inverse=reference, nuclear_repulsion=0.)


# 2. An external analytic formula; every parameter is explicit.
analytic = dft.custom_xc(
    input_fn=lambda fields: root@fields.density_matrix@root,
    energy_fn=lambda p, x: p['quadratic']*jnp.sum(x*x)
        +p['mixing']*jnp.sum(x[:, 0, 1]+x[:, 1, 0]),
    name='quadratic_demo')
trainer = training.Trainer(analytic, params={'quadratic': jnp.asarray(.1), 'mixing': jnp.asarray(.1)})
trainer.mode = 'fixed_density'
trainer.loss = {'energy': {'mse': 1., 'mae': 1.}, 'stationarity': {'norm2': .1}}
trainer.learning_rate = .005
trainer.run(sample, steps=20)
print('Analytic initial/final loss:', trainer.history['loss'][0], trainer.history['loss'][-1])
print('Analytic fitted parameters:', trainer.params)


# 3. A separately defined neural architecture uses exactly the same contract.
def network(params, x):
    hidden = jnp.tanh(x.reshape(-1)@params['w1']+params['b1'])
    return jnp.dot(hidden, params['w2'])+params['b2']


neural = dft.custom_xc(input_fn=lambda fields: root@fields.density_matrix@root,
                      energy_fn=network, name='external_MLP_demo')
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
params = dict(w1=.05*jax.random.normal(key1, (8, 4)), b1=jnp.zeros(4),
              w2=.05*jax.random.normal(key2, (4,)), b2=jnp.asarray(0.))
nn_trainer = training.Trainer(neural, params=params)
nn_trainer.loss = trainer.loss
nn_trainer.learning_rate = .005
nn_trainer.run(sample, steps=20)
print('Neural initial/final loss:', nn_trainer.history['loss'][0], nn_trainer.history['loss'][-1])


# 4. The agent receives the same raw, differentiable objective and metrics.
# Choosing a reward or updating an RL policy belongs to the external agent.
energy = iks.energy_function(state, analytic, layout='uks', nuclear_repulsion=0.)
value, score = iks.loss(reference, energy, trainer.params, weights=trainer.loss)
metrics = dict(loss=float(value), energy_error_hartree=float(score.energy_error),
               rotation_norm2=float(score.rotation_norm2), valid=bool(score.valid),
               reference_method=reference.method)
print('Agent metrics:', metrics)
print('Candidate SCF cycles:', int(trainer.metrics['scf_cycles']))

# Local CPU/float64, seed 0, 20 Adam updates per model:
# Analytic loss: 0.136762949 -> 0.0257815184
# Neural loss:   0.0972634234 -> 0.00766526941
# Analytic final energy error: -0.0251212104 Ha
# Analytic final rotation norm squared: 0.000292327828
# Candidate SCF cycles: 0. This is an interface check, not a converged fit.
