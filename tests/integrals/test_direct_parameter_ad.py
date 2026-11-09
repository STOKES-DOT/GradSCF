"""Direct geometry/basis first-response contract, Bohr and float64.

Dense ERIs appear only in the independent tiny oracle. Initial RED separates
supported cutoff-zero coordinates from missing screened/basis products.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


@pytest.fixture(scope="module")
def problem():
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H .2 -.1 1.4",
        {"H": [[0, [1.3, .7, .2], [.35, 0., .6]],
               [1, [.9, .6], [.25, .4]]]}, unit="Bohr", cart=True)
    plan = integrals.make_plan(topology)
    rng = np.random.default_rng(20261109)
    density = jnp.asarray(rng.normal(size=(2, topology.nao, topology.nao)))
    return plan, parameters, density


def changed(parameters, field, value):
    if field == "centers":
        return replace(parameters, centers=value)
    if field == "coefficients":
        return replace(parameters, coefficients=(value, *parameters.coefficients[1:]))
    exponents = list(parameters.exponents)
    exponents[1] = exponents[1] * jnp.exp(value)
    return replace(parameters, exponents=tuple(exponents))


def point_and_direction(parameters, field):
    if field == "centers":
        value = parameters.centers
        return value, .07*jnp.sin(jnp.arange(value.size).reshape(value.shape)+.2)
    if field == "coefficients":
        return parameters.coefficients[0], jnp.array([[.11, -.07], [.13, .05]])
    return jnp.zeros_like(parameters.exponents[1]), jnp.array([.08, -.04])


@pytest.mark.parametrize("field", ["centers", "coefficients", "log_exponents"])
@pytest.mark.parametrize("cutoff", [0., 1e-12])
@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_direct_first_parameter_products_match_small_eri_oracle(problem, field, cutoff, mode):
    plan, parameters, density = problem
    point, direction = point_and_direction(parameters, field)
    weights = (jnp.sin(density), .3*jnp.cos(density))

    def reference(value):
        eri = plan.evaluate("eri", changed(parameters, field, value))
        return (jnp.einsum("pqrs,brs->bpq", eri, density),
                jnp.einsum("prqs,brs->bpq", eri, density))

    def direct(value):
        return plan.get_jk(changed(parameters, field, value), density,
                           workers=2, screening_threshold=cutoff)

    def loss(operation, value):
        return sum(jnp.sum(x*w) for x, w in zip(operation(value), weights))

    if mode == "jvp":
        expected = jax.jit(lambda p, v: jax.jvp(reference, (p,), (v,))[1])(point, direction)
        actual = jax.jit(lambda p, v: jax.jvp(direct, (p,), (v,))[1])(point, direction)
    else:
        expected = jax.jit(jax.grad(lambda p: loss(reference, p)))(point)
        if field == "coefficients":
            assert abs(float(expected[1, 0])) > 1e-6
        actual = jax.jit(jax.grad(lambda p: loss(direct, p)))(point)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(got)).all()
        np.testing.assert_allclose(got, want, atol=3e-10, rtol=3e-9)
