"""Positive-cutoff support must retain active zero-coefficient basis response.

Coordinates are Bohr, exponents Bohr^-2, float64. An AO's compact primitive
is tiny in this block; its *active* diffuse primitive has coefficient zero.
The off-diagonal density makes its coefficient derivative physically visible
although its primal AO value is screened by an amplitude-only envelope.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids.blocks import AOGrid, _LocalAOGrid


def _raw(coefficient):
    return AOGrid(
        coords=jnp.asarray([[7.9, .1, -.1], [8.1, -.2, .2], [8., .2, .1]]),
        weights=jnp.asarray([.2, .3, .15]),
        centers=jnp.asarray([[0., 0., 0.], [8., 0., 0.], [-24., 0., 0.]]),
        exponents=jnp.asarray([[1., .02], [.8, 0.], [1.2, 0.]]),
        coefficients=jnp.asarray([[1., 0.], [1., 0.], [.8, 0.]]).at[0, 1].set(coefficient),
        nprims=(2, 1, 1), angulars=((0, 0, 0),)*3, block_size=2,
    )


def _density():
    return jnp.asarray([[1., .3, 0.], [.3, .9, 0.], [0., 0., .6]])


def _energy(coefficient, density, *, local, deriv):
    raw = _raw(coefficient)

    def consume(energy, ao, derivative, laplacian, coords, weights, indices):
        del coords
        dm = density[indices[:, None], indices[None, :]]
        rho = jnp.einsum("ga,ab,gb->g", ao, dm, ao)
        integrand = .5*rho**2
        if deriv:
            gradient = 2*jnp.einsum("xga,ab,gb->xg", derivative[1:], dm, ao)
            integrand = integrand + .02*jnp.sum(gradient**2, axis=0)
        if deriv == 2:
            lap = 2*(jnp.einsum("ga,ab,gb->g", laplacian, dm, ao)
                     + jnp.einsum("xga,ab,xgb->g", derivative[1:], dm, derivative[1:]))
            integrand = integrand + .003*lap**2
        return energy + jnp.dot(weights, integrand)

    initial = jnp.asarray(0., dtype=raw.coords.dtype)
    if local:
        provider = _LocalAOGrid.from_grid(raw, cutoff=1e-10, capacity=2)
        return provider.reduce_local(consume, initial, deriv=deriv)
    indices = jnp.arange(raw.nao)
    return raw.reduce(lambda energy, *block: consume(energy, *block, indices),
                      initial, deriv=deriv)


def _energy_and_field(coefficient, *, local, deriv):
    function = lambda dm: _energy(coefficient, dm, local=local, deriv=deriv)
    return jax.value_and_grad(function)(_density())


@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_zero_active_coefficient_keeps_energy_and_field_parameter_response(deriv):
    coefficient = jnp.asarray(0., dtype=jnp.float64)
    direction = jnp.asarray(1., dtype=jnp.float64)
    dense = lambda c: _energy_and_field(c, local=False, deriv=deriv)
    local = lambda c: _energy_and_field(c, local=True, deriv=deriv)
    expected, tangent = jax.jit(lambda c: jax.jvp(dense, (c,), (direction,)))(coefficient)
    actual, local_tangent = jax.jit(lambda c: jax.jvp(local, (c,), (direction,)))(coefficient)
    assert abs(float(tangent[0])) > 1e-6
    assert np.max(np.abs(np.asarray(tangent[1]))) > 1e-5
    for value in jax.tree.leaves((expected, tangent, actual, local_tangent)):
        assert np.isfinite(np.asarray(value)).all()
    for value, reference in zip(jax.tree.leaves((actual, local_tangent)),
                                jax.tree.leaves((expected, tangent))):
        np.testing.assert_allclose(value, reference, atol=2e-11, rtol=2e-10,
                                   equal_nan=False)
    for step in (1e-4, 5e-5):
        dense_fd = jax.tree.map(lambda high, low: (high-low)/(2*step),
                                dense(coefficient+step), dense(coefficient-step))
        local_fd = jax.tree.map(lambda high, low: (high-low)/(2*step),
                                local(coefficient+step), local(coefficient-step))
        for finite_difference in (dense_fd, local_fd):
            for value, reference in zip(jax.tree.leaves(finite_difference),
                                        jax.tree.leaves(tangent)):
                np.testing.assert_allclose(value, reference, atol=2e-9, rtol=2e-7,
                                           equal_nan=False)
