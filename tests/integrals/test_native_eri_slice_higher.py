"""Static Cartesian ERI windows retain cubic/quartic coordinate products."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


def _directional(function, coordinates, directions):
    current = function
    for direction in directions:
        previous = current
        current = lambda r, previous=previous, direction=direction: jax.jvp(
            previous, (r,), (direction,))[1]
    return current(coordinates)


def _problem():
    top, params = integrals.prepare_basis("H .1 -.2 0; H .3 .1 .8", {
        "H": [[1, [1.1, .8, .1], [.3, -.2, .9]]]}, cart=True)
    plan = integrals.make_plan(top)
    coordinates = jnp.concatenate((params.nuclear_coords, params.centers))
    half = top.nao//2
    def sliced(r):
        p = replace(params, nuclear_coords=r[:2], centers=r[2:])
        return plan.evaluate("eri", p, shell_slice=(0, 1, 1, 2, 0, 1, 1, 2))
    def reference(r):
        p = replace(params, nuclear_coords=r[:2], centers=r[2:])
        return plan.evaluate("eri", p)[:half, half:, :half, half:]
    return sliced, reference, coordinates


@pytest.mark.parametrize("order", [3, 4])
def test_static_eri_slice_higher_products_match_full_selection_and_fd(order):
    value, reference, r = _problem()
    directions = jnp.asarray(np.random.default_rng(29).normal(size=(order, *r.shape)))*.1
    actual = jax.jit(lambda x, d: _directional(value, x, d))(r, directions)
    expected = _directional(reference, r, directions)
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-9)
    previous = lambda x: _directional(value, x, directions[:-1])
    h = 2e-4
    fd = (previous(r+h*directions[-1])-previous(r-h*directions[-1]))/(2*h)
    np.testing.assert_allclose(actual, fd, atol=3e-8, rtol=3e-7)


def test_static_eri_slice_quartic_adjoint_and_vector_transposes():
    value, _, r = _problem()
    rng = np.random.default_rng(31)
    directions = jnp.asarray(rng.normal(size=(4, *r.shape)))*.1
    cotangent = jnp.asarray(rng.normal(size=value(r).shape))
    product = jax.jit(lambda d: _directional(value, r, d))
    adjoint = lambda c: _directional(jax.grad(lambda x: jnp.vdot(value(x), c)),
                                    r, directions[1:])
    pulled = jax.jit(adjoint)(cotangent)
    np.testing.assert_allclose(jnp.vdot(pulled, directions[0]),
                               jnp.vdot(product(directions), cotangent), atol=3e-10)
    transposed = jax.jit(jax.grad(lambda d: jnp.vdot(
        product(directions.at[0].set(d)), cotangent)))(directions[0])
    np.testing.assert_allclose(transposed, pulled, atol=3e-10, rtol=3e-9)
    np.testing.assert_allclose(jax.grad(lambda c: jnp.vdot(adjoint(c), directions[0]))(cotangent),
                               product(directions), atol=3e-10, rtol=3e-9)


def test_static_eri_slice_higher_ignores_unselected_high_angular_shells():
    atom = "H .1 -.2 0; H .3 .1 .8"
    top, params = integrals.prepare_basis(atom, {"H": [[0, [.9, 1.]], [11, [.7, 1.]]]}, cart=True)
    small_top, small_params = integrals.prepare_basis(atom, {"H": [[0, [.9, 1.]]]}, cart=True)
    plan, small_plan = integrals.make_plan(top), integrals.make_plan(small_top)
    r = jnp.concatenate((params.nuclear_coords, params.centers))
    def value(x):
        p = replace(params, nuclear_coords=x[:2], centers=x[2:])
        return plan.evaluate("eri", p, shell_slice=(0, 1, 2, 3, 0, 1, 2, 3)).reshape(())
    def reference(x):
        p = replace(small_params, nuclear_coords=x[:2], centers=x[jnp.array([2, 4])])
        return small_plan.evaluate("eri", p)[0, 1, 0, 1]
    directions = jnp.asarray(np.random.default_rng(37).normal(size=(4, *r.shape)))*.1
    np.testing.assert_allclose(jax.jit(lambda x, d: _directional(value, x, d))(r, directions),
                               _directional(reference, r, directions), atol=3e-10, rtol=3e-9)


def test_static_eri_slice_fifth_coordinate_derivative_fails_explicitly():
    value, _, r = _problem()
    directions = jnp.ones((5, *r.shape))
    with pytest.raises(NotImplementedError, match="(?i)fifth.*geometry"):
        _directional(value, r, directions)
