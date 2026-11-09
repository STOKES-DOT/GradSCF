"""Exact direct basis AD through provider preparation, in Bohr and float64.

Scope: ordinary Coulomb, cutoff=0, Cartesian general contractions, two real
nonsymmetric density batches. First-order checks cover raw coefficients and
log-exponent changes; mixed checks cover one geometry and one basis derivative.
They do not assert screened/RSH basis AD, pure basis Hessians, or SCF force loss.
The tiny dense oracle uses only supported first derivatives, including for FD.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals.plan import IntegralPlan
from test_direct_parameter_ad import changed, point_and_direction, problem


def _dense(plan, parameters, density):
    eri = plan.evaluate("eri", parameters)
    return (jnp.einsum("pqrs,brs->bpq", eri, density),
            jnp.einsum("prqs,brs->bpq", eri, density))


def _weighted(result, weights):
    return sum(jnp.sum(value*weight) for value, weight in zip(result, weights))


def _compare(actual, expected, *, atol=3e-10, rtol=3e-9):
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(got)).all()
        np.testing.assert_allclose(got, want, atol=atol, rtol=rtol)


def _check_first_products(action, reference, point, direction, weights):
    def products(operation, value, tangent):
        primal, pushed = jax.jvp(operation, (value,), (tangent,))
        pulled = jax.grad(lambda x: _weighted(operation(x), weights))(value)
        return primal, pushed, pulled

    actual = jax.jit(lambda x, v: products(action, x, v))
    expected = jax.jit(lambda x, v: products(reference, x, v))
    old = None
    for value in (point, point+.2*direction):
        want = expected(value, direction)
        got = actual(value, direction)
        _compare(got, want)
        assert float(jnp.linalg.norm(want[2])) > 1e-6
        np.testing.assert_allclose(_weighted(got[1], weights),
                                   jnp.vdot(got[2], direction), atol=3e-10, rtol=3e-9)
        if old is not None:
            assert any(float(jnp.linalg.norm(a-b)) > 1e-6
                       for a, b in zip(want[0], old)), "Updated basis must change J/K"
        old = want[0]


@pytest.mark.parametrize("field", ["coefficients", "log_exponents"])
def test_rebound_prepared_provider_retains_basis_jvp_vjp(problem, field):
    plan, parameters, density = problem
    point, direction = point_and_direction(parameters, field)
    weights = (jnp.sin(density), .3*jnp.cos(density))
    original = NativeDirectBasis(plan, parameters)

    def action(value):
        # dataclasses.replace retains the original derived preparation operand;
        # the provider must notice the new raw state and rebind its live ENV.
        provider = replace(original, parameters=changed(parameters, field, value))
        return provider.get_jk(density, workers=2, screening_threshold=0.)

    reference = lambda value: _dense(plan, changed(parameters, field, value), density)
    _check_first_products(action, reference, point, direction, weights)


@pytest.mark.parametrize("field", ["coefficients", "log_exponents"])
def test_raw_provider_pytree_update_retains_basis_jvp_vjp(problem, field):
    plan, parameters, density = problem
    point, direction = point_and_direction(parameters, field)
    weights = (jnp.sin(density), .3*jnp.cos(density))
    provider = NativeDirectBasis(plan, parameters)
    leaves, structure = jax.tree.flatten(provider)
    target = parameters.coefficients[0] if field == "coefficients" else parameters.exponents[1]
    slots = [index for index, value in enumerate(leaves) if value is target]
    assert len(slots) == 1, "Provider PyTree must expose the canonical raw basis leaf"
    consume = jax.jit(lambda basis: basis.get_jk(density, workers=2, screening_threshold=0.))

    def action(value):
        updated = list(leaves)
        updated[slots[0]] = value if field == "coefficients" else target*jnp.exp(value)
        return consume(jax.tree.unflatten(structure, updated))

    reference = lambda value: _dense(plan, changed(parameters, field, value), density)
    _check_first_products(action, reference, point, direction, weights)


@pytest.mark.parametrize("field", ["coefficients", "log_exponents"])
def test_prepared_response_with_s8_cache_budget_retains_basis_jvp_vjp(problem, field):
    plan, parameters, density = problem
    point, direction = point_and_direction(parameters, field)
    weights = (jnp.sin(density), .3*jnp.cos(density))

    def action(value):
        provider = NativeDirectBasis(plan, changed(parameters, field, value))
        # Cache admission must not freeze an active basis derivative. The API
        # may select its bounded direct action for traced parameters.
        response = provider.prepare_jk(cache_bytes=2*provider.eri_cache_nbytes,
                                       workers=2, screening_threshold=0.)
        return response(density)

    reference = lambda value: _dense(plan, changed(parameters, field, value), density)
    _check_first_products(action, reference, point, direction, weights)


@pytest.mark.parametrize("field", ["coefficients", "log_exponents"])
def test_bound_geometry_basis_mixed_products_match_first_gradient_fd(problem, field, monkeypatch):
    plan, parameters, density = problem
    point, direction = point_and_direction(parameters, field)
    centers, geometry_direction = point_and_direction(parameters, "centers")
    weights = (jnp.sin(density), .3*jnp.cos(density))

    def loss(r, value, *, dense):
        current = replace(changed(parameters, field, value), centers=r)
        result = (_dense(plan, current, density) if dense else
                  NativeDirectBasis(plan, current).get_jk(
                      density, workers=2, screening_threshold=0.))
        return _weighted(result, weights)

    # Differentiate only geometry here. Parameters remain fixed operands of
    # each exact first derivative; no unsupported dense mixed AD is requested.
    exact_geometry = jax.jit(jax.grad(lambda r, x: loss(r, x, dense=True), argnums=0))
    references = []
    for step in (2e-4, 1e-4):
        plus = exact_geometry(centers, point+step*direction)
        minus = exact_geometry(centers, point-step*direction)
        pushed = (plus-minus)/(2*step)
        pulled = []
        for index in range(point.size):
            unit = jnp.zeros_like(point).reshape(-1).at[index].set(1.).reshape(point.shape)
            delta = (exact_geometry(centers, point+step*unit)
                     - exact_geometry(centers, point-step*unit))/(2*step)
            pulled.append(jnp.vdot(geometry_direction, delta))
        references.append((pushed, jnp.stack(pulled).reshape(point.shape)))
    _compare(references[0], references[1], atol=2e-7, rtol=4e-6)
    assert float(jnp.linalg.norm(references[1][0])) > 1e-6
    assert float(jnp.linalg.norm(references[1][1])) > 1e-6

    evaluate = IntegralPlan.evaluate

    def forbid_eri(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("Direct mixed products must not construct AO ERIs")
        return evaluate(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", forbid_eri)
    geometry = jax.grad(lambda r, x: loss(r, x, dense=False), argnums=0)
    pushed = jax.jit(lambda x, v: jax.jvp(
        lambda y: geometry(centers, y), (x,), (v,))[1])(point, direction)
    pulled = jax.jit(jax.grad(lambda x: jnp.vdot(
        geometry_direction, geometry(centers, x))))(point)
    for expected in references:
        _compare((pushed, pulled), expected, atol=2e-7, rtol=4e-6)
    np.testing.assert_allclose(jnp.vdot(geometry_direction, pushed),
                               jnp.vdot(pulled, direction), atol=2e-9, rtol=2e-8)
