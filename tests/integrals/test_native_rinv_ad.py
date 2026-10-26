"""Native point/Gaussian Rinv products; CPU float64, Bohr and Bohr**-2."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.autodiff.evaluation import evaluate_differentiable


def _problem(cart, zeta):
    top, params = integrals.prepare_basis("H 0 0 0; He .2 .1 .8", {
        "H": [[0, [1.1, .8, -.2], [.3, .2, .9]], [2, [.7, 1.]]],
        "He": [[0, [.8, 1.]], [1, [.5, 1.]]]}, cart=cart)
    plan = integrals.make_plan(top)
    fixed = replace(params, centers=jnp.zeros_like(params.centers),
                    nuclear_coords=jnp.zeros_like(params.nuclear_coords))
    atm, bas, env, coefficients, exponents = plan.coefficient_data(fixed, separate_exponents=True)
    env = env.at[7].set(zeta)
    geometry = jnp.concatenate((params.nuclear_coords, params.centers, jnp.array([[.3, -.2, .1]])))
    def value(c, a, r):
        return evaluate_differentiable("rinv", atm, bas, env, c, r[:-1], r[-1],
                                       top.nao, cart=cart, exponents=a)
    return value, coefficients, exponents, geometry


def _fd(function, x, direction, h=2e-4):
    return (8*(function(x+h*direction)-function(x-h*direction))
            -function(x+2*h*direction)+function(x-2*h*direction))/(12*h)


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("zeta", [0., .16])
def test_rinv_coordinate_and_origin_first_second_products(cart, zeta):
    value, c, a, r = _problem(cart, zeta)
    function = lambda x: value(c, a, x)
    rng = np.random.default_rng(84)
    u, v = [jnp.asarray(rng.normal(size=r.shape))*.1 for _ in range(2)]
    primal, tangent = jax.jit(lambda x, d: jax.jvp(function, (x,), (d,)))(r, v)
    np.testing.assert_allclose(tangent, _fd(function, r, v), atol=3e-9, rtol=3e-7)
    cotangent = jnp.asarray(rng.normal(size=primal.shape))
    gradient = jax.jit(jax.grad(lambda x: jnp.vdot(function(x), cotangent)))
    np.testing.assert_allclose(jnp.vdot(gradient(r), v), jnp.vdot(cotangent, tangent), atol=3e-10)
    first = lambda x: jax.jvp(function, (x,), (v,))[1]
    second = jax.jit(lambda x: jax.jvp(first, (x,), (u,))[1])(r)
    np.testing.assert_allclose(second, _fd(first, r, u), atol=3e-8, rtol=3e-7)
    hvp = jax.jit(lambda x: jax.jvp(gradient, (x,), (v,))[1])(r)
    np.testing.assert_allclose(hvp, _fd(gradient, r, v), atol=3e-8, rtol=3e-7)
    np.testing.assert_allclose(jnp.vdot(hvp, u), jnp.vdot(cotangent, second), atol=3e-10)
    for axis in range(3):
        translation = jnp.zeros_like(r).at[:, axis].set(1.)
        np.testing.assert_allclose(jax.jvp(function, (r,), (translation,))[1], 0., atol=3e-12)


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("zeta", [0., .16])
def test_rinv_coefficient_exponent_first_products(cart, zeta):
    value, c, a, r = _problem(cart, zeta)
    x = jnp.concatenate((c, a))
    direction = jnp.sin(jnp.arange(x.size)+1)*.03
    function = lambda x: value(x[:c.size], x[c.size:], r)
    primal, tangent = jax.jit(lambda x, d: jax.jvp(function, (x,), (d,)))(x, direction)
    np.testing.assert_allclose(tangent, _fd(function, x, direction), atol=3e-9, rtol=3e-7)
    cotangent = jnp.cos(jnp.arange(primal.size)).reshape(primal.shape)
    gradient = jax.jit(jax.grad(lambda x: jnp.vdot(function(x), cotangent)))(x)
    np.testing.assert_allclose(jnp.vdot(gradient, direction), jnp.vdot(cotangent, tangent), atol=3e-10)
    coefficient_gradient = jax.grad(lambda c: jnp.vdot(value(c, a, r), cotangent))
    d = direction[:c.size]
    np.testing.assert_allclose(jax.jvp(coefficient_gradient, (c,), (d,))[1],
                               _fd(coefficient_gradient, c, d), atol=3e-9, rtol=3e-7)
