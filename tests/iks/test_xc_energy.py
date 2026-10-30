"""Explicit XC-energy labels remain separate from total-energy targets."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks


def target():
    return iks.Target(jnp.array([[2.0]]), jnp.eye(1), nelec=2, layout="rks")


def test_xc_energy_loss_accepts_an_explicit_xc_label_without_total_energy():
    value, score = iks.xc_energy_loss(
        target(), lambda params, density: params * jnp.trace(density), 0.4,
        target_xc_energy=0.5, weights={"mse": 1.0, "mae": 1.0}, scale=1.0,
    )

    np.testing.assert_allclose(score.xc_energy, 0.8, atol=2e-13)
    np.testing.assert_allclose(score.error, 0.3, atol=2e-13)
    np.testing.assert_allclose(value, 0.39, atol=2e-13)
    assert score.valid


def test_xc_energy_loss_requires_a_separate_label_and_rejects_bad_scale():
    with pytest.raises(ValueError, match="XC|label"):
        iks.xc_energy_loss(target(), lambda params, density: params, 0.4)
    with pytest.raises(ValueError, match="scale"):
        iks.xc_energy_loss(target(), lambda params, density: params, 0.4,
                           target_xc_energy=0.5, scale=0.0)
