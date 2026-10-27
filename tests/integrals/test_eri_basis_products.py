"""Full libcint ERI basis products: float64, Bohr, atomic units, real FD oracles."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


def _problem(cart):
    # General s contractions, an individual zero coefficient, and a d shell
    # exercise shadow contractions, angular raising and the spherical transform.
    topology, parameters = integrals.prepare_basis("He 0 0 0", {
        "He": [[0, [1.3, .7, .2], [.35, 0., .6]],
               [2, [.9, .6], [.25, .4]]],
    }, unit="Bohr", cart=cart)
    plan = integrals.make_plan(topology)
    weights = jnp.sin(jnp.arange(topology.nao**4, dtype=jnp.float64)+.3).reshape((topology.nao,)*4)
    return plan, parameters, weights


@pytest.mark.parametrize("cart", [True, False])
def test_full_eri_raw_coefficient_and_exponent_jvp_vjp_match_fd(cart):
    plan, parameters, weights = _problem(cart)
    coefficients, exponents = parameters.coefficients[0], parameters.exponents[1]
    dc = jnp.asarray([[.11, -.07], [.13, .05]])
    da = jnp.asarray([.09, -.06])

    def tensor(c, a):
        moving = replace(parameters,
            coefficients=(c, parameters.coefficients[1]),
            exponents=(parameters.exponents[0], a))
        return plan.evaluate("eri", moving)

    value, tangent = jax.jit(lambda c, a: jax.jvp(tensor, (c, a), (dc, da)))(coefficients, exponents)
    _, pullback = jax.vjp(tensor, coefficients, exponents)
    gc, ga = pullback(weights)
    np.testing.assert_allclose(jnp.vdot(weights, tangent),
        jnp.vdot(gc, dc)+jnp.vdot(ga, da), atol=2e-10, rtol=2e-10)
    assert float(jnp.max(jnp.abs(gc))) > 1e-5
    # The zero individual raw coefficient still has a nonzero derivative.
    assert abs(float(gc[1, 0])) > 1e-5
    for step in (2e-4, 1e-4):
        fd = (tensor(coefficients+step*dc, exponents+step*da)
              -tensor(coefficients-step*dc, exponents-step*da))/(2*step)
        np.testing.assert_allclose(tangent, fd, atol=3e-8, rtol=3e-6)
    assert np.isfinite(np.asarray(value)).all()


@pytest.mark.parametrize("cart", [True, False])
def test_full_eri_coefficient_hvp_matches_parameter_gradient_fd(cart):
    plan, parameters, weights = _problem(cart)
    coefficients = parameters.coefficients[0]
    direction = jnp.asarray([[.11, -.07], [.13, .05]])

    def energy(c):
        moving = replace(parameters, coefficients=(c, parameters.coefficients[1]))
        return jnp.vdot(weights, plan.evaluate("eri", moving))

    gradient = jax.jit(jax.grad(energy))
    hvp = jax.jit(lambda c: jax.jvp(gradient, (c,), (direction,))[1])(coefficients)
    for step in (2e-4, 1e-4):
        fd = (gradient(coefficients+step*direction)-gradient(coefficients-step*direction))/(2*step)
        np.testing.assert_allclose(hvp, fd, atol=2e-7, rtol=4e-6)


def test_full_eri_higher_exponent_products_fail_explicitly():
    plan, parameters, weights = _problem(True)

    def energy(a):
        moving = replace(parameters, exponents=(parameters.exponents[0], a))
        return jnp.vdot(weights, plan.evaluate("eri", moving))

    with pytest.raises(NotImplementedError, match="higher|mixed|exponent"):
        jax.jvp(jax.grad(energy), (parameters.exponents[1],), (jnp.ones(2),))


def test_full_eri_dynamic_basis_keeps_third_and_fourth_coordinate_products():
    from gradscf.integrals.backends.native.autodiff.geometry import evaluate_geometry

    topology, parameters = integrals.prepare_basis("H 0 0 0; H .1 -.2 1.4",
        {"H": [[0, [.9, 1.]]]}, unit="Bohr")
    plan = integrals.make_plan(topology)
    direction = jnp.asarray([[.07, -.03, .09], [-.02, .04, -.05]])
    weights = jnp.cos(jnp.arange(16., dtype=jnp.float64)).reshape((2,)*4)

    def bundle(p, canonical):
        def value(centers):
            moving = replace(p, centers=centers)
            if canonical:
                tensor = plan.evaluate("eri", moving)
            else:
                fixed = replace(moving, centers=jnp.zeros_like(centers),
                                nuclear_coords=jnp.zeros_like(p.nuclear_coords))
                atoms, basis, env = plan.pack(fixed)
                coords = jnp.concatenate((p.nuclear_coords, centers))
                tensor = evaluate_geometry("eri", atoms, basis, env, coords,
                    jnp.zeros(3), topology.nao, cart=True)
            return jnp.vdot(weights, tensor)

        def directional(function):
            return lambda r: jax.jvp(function, (r,), (direction,))[1]

        third = directional(directional(directional(value)))
        fourth = directional(third)
        return third(p.centers), fourth(p.centers)

    actual = jax.jit(lambda p: bundle(p, True))(parameters)
    expected = jax.jit(lambda p: bundle(p, False))(parameters)
    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-10)


@pytest.mark.parametrize("cart", [True, False])
def test_shell_selected_eri_basis_products_match_full_selection(cart):
    plan, parameters, _ = _problem(cart)
    bounds = (0, 1, 1, 2, 0, 2, 1, 2)
    ns = 2
    nd = 6 if cart else 5
    selection = (slice(0, ns), slice(ns, ns+nd), slice(None), slice(ns, ns+nd))
    c, a = parameters.coefficients[0], parameters.exponents[1]
    dc, da = jnp.asarray([[.11, -.07], [.13, .05]]), jnp.asarray([.09, -.06])

    def tensor(c, a, bounded):
        moving = replace(parameters, coefficients=(c, parameters.coefficients[1]),
                         exponents=(parameters.exponents[0], a))
        if bounded:
            return plan.evaluate("eri", moving, shell_slice=bounds)
        return plan.evaluate("eri", moving)[selection]

    actual, tangent = jax.jit(lambda c, a: jax.jvp(
        lambda x, y: tensor(x, y, True), (c, a), (dc, da)))(c, a)
    expected, expected_tangent = jax.jit(lambda c, a: jax.jvp(
        lambda x, y: tensor(x, y, False), (c, a), (dc, da)))(c, a)
    assert actual.shape == (ns, nd, ns+nd, nd)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(tangent, expected_tangent, atol=3e-11, rtol=3e-10)
    weights = jnp.cos(jnp.arange(actual.size, dtype=jnp.float64)).reshape(actual.shape)
    _, pullback = jax.vjp(lambda x, y: tensor(x, y, True), c, a)
    gc, ga = pullback(weights)
    np.testing.assert_allclose(jnp.vdot(weights, tangent), jnp.vdot(gc, dc)+jnp.vdot(ga, da),
                               atol=3e-10, rtol=3e-10)
