"""Analytic DF-RHF orbital curvature matches the Cayley-energy AD definition."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.scf.orbital_optimization import _rotation_functions


@pytest.mark.parametrize('nocc', [1, 3, 5])
def test_df_rhf_curvature_matches_energy_hessian_away_from_stationarity(nocc):
    from gradscf.scf.stability import _df_rhf_orbital_curvature
    rng = np.random.default_rng(18)
    n = 6
    c = jnp.asarray(np.linalg.qr(rng.normal(size=(n, n)))[0])
    # Nontrivial AO metric: C need not be Euclidean orthonormal.
    c = jnp.diag(jnp.linspace(.7, 1.4, n))@c
    h = jnp.asarray(rng.normal(size=(n, n)))
    h = (h+h.T)/2
    b = jnp.asarray(rng.normal(size=(8, n, n))*.1)
    b = (b+b.swapaxes(-1, -2))/2
    occ = (1.,)*nocc+(0.,)*(n-nocc)
    dimension, rotate, density = _rotation_functions('roks', (occ, occ))
    def energy(x):
        d = jnp.sum(density(rotate(x, c[None])), axis=0)
        j, k = build_jk_from_df(b, d)
        return jnp.sum(d*h)+.5*jnp.sum(d*j)-.25*jnp.sum(d*k)
    origin = jnp.zeros(dimension)
    expected_g = jax.grad(energy)(origin)
    expected_h = jax.hessian(energy)(origin)
    gradient, apply = _df_rhf_orbital_curvature(h, b, c, nocc)
    actual_h = jax.jit(jax.vmap(apply, in_axes=1, out_axes=1))(jnp.eye(dimension))
    np.testing.assert_allclose(gradient, expected_g, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual_h, expected_h, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(actual_h, actual_h.T, atol=2e-12, rtol=0)
    direction = jnp.asarray(rng.normal(size=dimension))
    step = 1e-5
    fd = (jax.grad(energy)(step*direction)-jax.grad(energy)(-step*direction))/(2*step)
    np.testing.assert_allclose(apply(direction), fd, atol=1e-7, rtol=1e-7)
