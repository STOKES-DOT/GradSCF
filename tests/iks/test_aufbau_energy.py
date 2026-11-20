"""Stationary energy gradients reuse the shared spectral-sum response."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.dft.iks.reward import _aufbau


def test_optional_energy_is_the_occupied_spectral_sum_including_degeneracy():
    h = jnp.diag(jnp.array([-2., -2., 1., 1.]))
    projector, gap, valid, energy = _aufbau(h, 2, 1e-7, return_energy=True)
    assert valid
    np.testing.assert_allclose(energy, -4., atol=1e-14)
    np.testing.assert_allclose(energy, jnp.trace(h@projector), atol=1e-14)
    direction = jnp.ones((4, 4))*.03
    objective = lambda p: _aufbau(h+p*direction, 2, 1e-7, return_energy=True)[3]
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(0.), jnp.trace(direction@projector), atol=1e-13)
    assert np.isfinite(jax.jit(jax.grad(jax.grad(objective)))(0.))


def test_empty_and_full_occupied_spaces_return_consistent_energies():
    h = jnp.diag(jnp.array([-2., 1.]))
    for count, expected in [(0, 0.), (2, -1.)]:
        _, _, valid, energy = _aufbau(h, count, 1e-7, return_energy=True)
        assert valid
        np.testing.assert_allclose(energy, expected, atol=1e-14)
