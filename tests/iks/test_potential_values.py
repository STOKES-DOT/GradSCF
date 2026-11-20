"""Sampled local fields must share the physical AO projection and its AD."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def grid():
    from gradscf.integrals.grids.blocks import AOGrid
    return AOGrid(jnp.array([[.1*i, .2, .07*i] for i in range(7)]),
        jnp.linspace(.1, .3, 7), jnp.array([[0., 0., 0.], [.5, 0., .1]]),
        jnp.array([[.7], [1.1]]), jnp.ones((2, 1)), (1, 1), ((0, 0, 0),)*2, 3)


def test_scalar_and_multiple_fields_match_direct_quadrature():
    from gradscf.dft.iks import project_potential
    provider = grid()
    ao = provider.block(0, provider.ngrids)[0]
    fields = jnp.stack((provider.coords[:,0], provider.coords[:,1]**2), axis=1)
    expected = jnp.einsum('g,gi,gj,gc->cij', provider.weights, ao, ao, fields)
    np.testing.assert_allclose(project_potential(provider, fields), expected, atol=1e-13)
    np.testing.assert_allclose(project_potential(provider, fields[:,0]), expected[0], atol=1e-13)


def test_projection_preserves_field_and_geometry_jvps():
    from gradscf.dft.iks import project_potential
    provider = grid()
    fields = provider.coords[:,0]
    def objective(values, centers):
        return jnp.sum(project_potential(replace(provider, centers=centers), values))
    direction = jnp.ones_like(fields)*.03
    centers = jnp.ones_like(provider.centers)*.02
    _, tangent = jax.jvp(objective, (fields, provider.centers), (direction, centers))
    step = 1e-4
    fd = (objective(fields+step*direction, provider.centers+step*centers)-
          objective(fields-step*direction, provider.centers-step*centers))/(2*step)
    np.testing.assert_allclose(tangent, fd, atol=1e-10, rtol=1e-7)


def test_bad_sampled_fields_are_rejected():
    from gradscf.dft.iks import project_potential
    with pytest.raises(ValueError): project_potential(grid(), jnp.ones(6))
    with pytest.raises(ValueError): project_potential(grid(), jnp.ones(7, dtype=complex))
    with pytest.raises(ValueError): project_potential(grid(), jnp.full(7,jnp.nan))
