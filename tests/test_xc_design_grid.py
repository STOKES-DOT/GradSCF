"""The molecule adapter must preserve the saved spherical variational space."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.integrals.grids.blocks import AOGrid


def test_projected_grid_density_matches_the_cartesian_embedding():
    from gradscf.model.autoxc.workflows.xc100.molecule import SphericalGrid
    grid = AOGrid(jnp.array([[.1*i, .2, .07*i] for i in range(7)]),
        jnp.linspace(.1, .3, 7), jnp.array([[0., 0., 0.], [.5, 0., .1]]),
        jnp.array([[.7], [1.1]]), jnp.ones((2, 1)), (1, 1), ((0, 0, 0),)*2, 3)
    transform = jnp.array([[1.], [.5]])
    projected = SphericalGrid(grid, transform)
    def integral(p):
        def add(total, ao, dao, lap, coords, weights):
            return total+jnp.sum(weights*jnp.einsum('gi,ij,gj->g', ao, p*jnp.eye(1), ao))
        return projected.reduce(add, jnp.asarray(0.))
    ao = grid.block(0, grid.ngrids)[0]
    expected = jnp.sum(grid.weights*jnp.einsum('gi,ij,gj->g', ao, transform@transform.T, ao))
    np.testing.assert_allclose(integral(1.), expected, atol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(integral))(1.), expected, atol=1e-14)
