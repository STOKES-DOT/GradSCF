"""Handwritten external PBE must agree with the frozen independent libxc table."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def test_handwritten_pbe_energy_and_density_derivative():
    from gradscf_tools.xc_design.pbe import energy_density
    table = json.loads((Path(__file__).parent/'fixtures/pbe_reference_table.json').read_text())
    for row in table['points']:
        n, sigma = row['density'], row['sigma']
        actual = energy_density(n, sigma, floor=0.)
        np.testing.assert_allclose(actual, row['energy_per_volume'], rtol=1e-12, atol=1e-15)
        derivative = jax.grad(energy_density, argnums=0)(jnp.asarray(n), jnp.asarray(sigma), floor=0.)
        np.testing.assert_allclose(derivative, row['de_dn'], rtol=1e-8, atol=1e-12)


def test_handwritten_pbe_gradient_terms_and_low_density_are_finite():
    from gradscf_tools.xc_design.pbe import energy_density
    table = json.loads((Path(__file__).parent/'fixtures/pbe_reference_table.json').read_text())
    for row in table['points']:
        if row['de_dsigma'] is not None:
            actual = jax.grad(energy_density, argnums=1)(
                jnp.asarray(row['density']), jnp.asarray(row['sigma']), floor=0.)
            np.testing.assert_allclose(actual, row['de_dsigma'], rtol=1e-5, atol=1e-11)
    value, gradient = jax.value_and_grad(lambda n: energy_density(n, n*n))(.0)
    assert np.isfinite([value, gradient]).all()
