"""Real zero-weight points retain weight response; only tail points are padding.

Coordinates are Bohr, exponents Bohr^-2, and all arrays use float64. The AO
at the real zero-weight point is far above the 1e-12 cutoff. Its weight is an
independent scalar, so the two-sided derivative exists at exactly zero even
when the old weight-based support selection changes there.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids.blocks import AOGrid, _LocalAOGrid


def _raw(weight, layout):
    if layout == "single":
        coords = [[.2, -.1, .3]]
        centers = [[0., 0., 0.]]
        weights = jnp.asarray([weight], dtype=jnp.float64)
        block_size = 4
    else:
        # The real zero is an interior input point, not padding. Spatial order
        # puts it in the first three-point block, far from its nonzero points.
        # The final block has both genuine weights and one actual padded slot.
        coords = [[20.2, .1, .2], [.1, .1, .1], [10., .3, -.2],
                  [-.2, .1, .1], [20., -.1, .1]]
        centers = [[0., 0., 0.], [10., .3, -.2], [20., 0., 0.], [-30., 0., 0.]]
        weights = jnp.asarray([.25, .15, 0., -.04, -.12], dtype=jnp.float64)
        weights = weights.at[2].set(weight)
        block_size = 3
    nao = len(centers)
    return AOGrid(
        coords=jnp.asarray(coords, dtype=jnp.float64), weights=weights,
        centers=jnp.asarray(centers, dtype=jnp.float64),
        exponents=jnp.ones((nao, 1), dtype=jnp.float64),
        coefficients=jnp.ones((nao, 1), dtype=jnp.float64),
        nprims=(1,)*nao, angulars=((0, 0, 0),)*nao, block_size=block_size,
    )


def _energy_and_potential(weight, *, layout, deriv, local):
    raw = _raw(weight, layout)
    density = .8*jnp.eye(raw.nao) + .1*jnp.ones((raw.nao, raw.nao))
    initial = (jnp.asarray(0., dtype=jnp.float64), jnp.zeros_like(density))

    def consume(carry, ao, derivative, laplacian, coords, weights, indices):
        del coords
        features = ao
        if deriv:
            features = features + .02*jnp.sum(derivative[1:], axis=0)
        if deriv == 2:
            features = features + .03*laplacian
        dm = density[indices[:, None], indices[None, :]]
        rho = jnp.einsum("gi,ij,gj->g", features, dm, features)
        # V is the full-AO derivative of E = 1/2 sum_g w_g rho_g**2.
        potential = jnp.einsum("g,gi,gj->ij", weights*rho, features, features)
        return (carry[0] + .5*jnp.dot(weights, rho**2),
                carry[1].at[indices[:, None], indices[None, :]].add(potential))

    if local:
        provider = _LocalAOGrid.from_grid(raw, cutoff=1e-12,
                                          capacity=1 if layout == "single" else 3)
        return provider.reduce_local(consume, initial, deriv=deriv)
    indices = jnp.arange(raw.nao)
    return raw.reduce(lambda carry, *block: consume(carry, *block, indices),
                      initial, deriv=deriv)


def _close(actual, expected, *, atol=2e-11, rtol=2e-10):
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(value)).all()
        assert np.isfinite(np.asarray(reference)).all()
        np.testing.assert_allclose(value, reference, atol=atol, rtol=rtol,
                                   equal_nan=False)


@pytest.mark.parametrize("layout", ["single", "mixed_tail"])
@pytest.mark.parametrize("deriv", [0, 1, 2])
@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_real_zero_weight_energy_and_ao_potential_response(layout, deriv, mode):
    zero = jnp.asarray(0., dtype=jnp.float64)
    one = jnp.asarray(1., dtype=jnp.float64)
    raw = _raw(zero, layout)
    zero_index = 0 if layout == "single" else 2
    # Establish that the genuine zero-weight point has a large physical AO.
    assert float(jnp.max(jnp.abs(raw.block(zero_index, 1)[0]))) > .1
    assert raw.ngrids % raw.block_size != 0
    if layout == "mixed_tail":
        assert bool(jnp.any(raw.weights > 0)) and bool(jnp.any(raw.weights < 0))
        assert 0 < zero_index < raw.ngrids-1

    dense = jax.jit(lambda w: _energy_and_potential(
        w, layout=layout, deriv=deriv, local=False))
    local = jax.jit(lambda w: _energy_and_potential(
        w, layout=layout, deriv=deriv, local=True))
    expected, tangent = jax.jvp(dense, (zero,), (one,))
    _close(local(zero), expected)
    assert abs(float(tangent[0])) > 1e-3
    assert np.max(np.abs(np.asarray(tangent[1]))) > 1e-3

    # Both sides retain the AO. Agreement with the dense derivative proves
    # this is an ordinary derivative of the same forward function, not a
    # comparison of unrelated approximations across the AO cutoff.
    for step in (1e-4, 5e-5):
        for function in (dense, local):
            finite_difference = jax.tree.map(
                lambda high, low: (high-low)/(2*step), function(step), function(-step))
            _close(finite_difference, tangent, atol=2e-10, rtol=2e-9)

    if mode == "jvp":
        actual, actual_tangent = jax.jvp(local, (zero,), (one,))
        _close(actual, expected)
        _close(actual_tangent, tangent)
    else:
        potential_seed = .37*jnp.eye(raw.nao) + .11*jnp.ones((raw.nao, raw.nao))
        seeds = ((one, jnp.zeros_like(potential_seed)), (zero, potential_seed))
        _, dense_pullback = jax.vjp(dense, zero)
        _, local_pullback = jax.vjp(local, zero)
        for seed in seeds:
            expected_gradient = dense_pullback(seed)[0]
            actual_gradient = local_pullback(seed)[0]
            assert abs(float(expected_gradient)) > 1e-3
            _close(actual_gradient, expected_gradient)
