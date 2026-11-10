"""Two compact AO capacities preserve the active set and dense fallback.

Float64; coordinates in Bohr and Gaussian exponents in Bohr^-2. The frozen
AO cutoff is 1e-14; energy and directional derivatives use absolute 1e-10.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids.blocks import AOGrid, _LocalAOGrid


def _raw(active, parameters=None):
    p = jnp.zeros(6) if parameters is None else parameters
    rows = jnp.arange(880)
    # A coprime permutation distributes active rows across the entire AO basis.
    near = (37*rows) % 880 < active
    offsets = jnp.stack(((rows % 5)*.03, (rows % 7)*.02, (rows % 3)*.04), axis=1)
    centers = offsets + jnp.where(near, 0., 25.)[:, None]
    centers = centers + p[0]*jnp.asarray([.2, -.1, .3])
    exponents = (.75+.2*(rows % 7)/7)[:, None]*jnp.exp(p[1])
    coefficients = (.3+rows/1760)[:, None]*(1+p[2])
    coords = jnp.asarray([[.1, .3, -.2], [-.4, .2, .1], [.3, -.2, .5],
                          [.2, .1, -.3], [-.2, -.1, .2]])
    coords = coords + p[3]*jnp.asarray([.1, .2, -.2])
    weights = jnp.asarray([.2, .1, -.04, .3, .15])*jnp.exp(p[4])
    return AOGrid(coords, weights, centers, exponents, coefficients,
                  (1,)*880, ((0, 0, 0),)*880, 4)


@pytest.mark.parametrize("active,capacity,expected_width", [
    (100, 512, 256), (256, 512, 256), (257, 512, 512),
    (300, 512, 512), (512, 512, 512), (513, 512, 880), (600, 512, 880),
    (100, 255, 255), (128, 256, 128), (129, 256, 256),
])
def test_executed_compact_width_uses_smallest_bucket_and_retains_dense_overflow(
        active, capacity, expected_width):
    local = _LocalAOGrid.from_grid(_raw(active), cutoff=1e-14, capacity=capacity)

    def consume(carry, ao, derivative, laplacian, coords, weights, indices):
        assert indices.shape[0] == ao.shape[-1]
        assert derivative.shape[-1] == laplacian.shape[-1] == ao.shape[-1]
        # This is returned by the executed branch, not a Python tracing counter.
        checksum = jnp.sum(ao**2)+jnp.sum(derivative**2)+jnp.sum(laplacian**2)
        return (carry[0]+jnp.asarray(ao.shape[-1], dtype=jnp.int32), carry[1]+checksum)

    actual = jax.jit(lambda grid: grid.reduce_local(
        consume, (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.)), deriv=2))(local)
    assert int(actual[0]) == 2*expected_width  # full block and a padded final block
    assert np.isfinite(np.asarray(actual[1])).all() and float(actual[1]) > 0.


def _energy(parameters, active, *, local):
    raw = _raw(active, parameters)
    grid = _LocalAOGrid.from_grid(raw, cutoff=1e-14, capacity=512) if local else raw
    orbital = (.5+jnp.arange(raw.nao)/raw.nao)/jnp.sqrt(float(raw.nao))
    orbital = orbital*jnp.exp(parameters[5])

    def consume(carry, ao, derivative, laplacian, coords, weights, indices):
        features = ao+.02*jnp.sum(derivative[1:], axis=0)+.03*laplacian
        rho = (features @ orbital[indices])**2
        return carry+.5*jnp.dot(weights, rho**2)

    if local:
        return grid.reduce_local(consume, jnp.asarray(0.), deriv=2)
    return raw.reduce(lambda carry, *args: consume(
        carry, *args, jnp.arange(raw.nao)), jnp.asarray(0.), deriv=2)


@pytest.mark.parametrize("active", [100, 300, 600])
def test_bucket_energy_and_geometry_basis_density_jvp_match_dense(active):
    p = jnp.asarray([.03, -.02, .01, .04, -.03, .02])
    direction = jnp.asarray([.2, -.3, .1, .4, -.2, .3])
    results = []
    for use_local in (True, False):
        function = lambda x: _energy(x, active, local=use_local)
        results.append(jax.jit(lambda x: jax.jvp(function, (x,), (direction,)))(p))
    for value, reference in zip(*results):
        assert np.isfinite(np.asarray(value)).all()
        assert np.isfinite(np.asarray(reference)).all()
        np.testing.assert_allclose(value, reference, atol=1e-10, rtol=0.)
