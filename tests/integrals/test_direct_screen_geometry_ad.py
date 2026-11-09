"""Screened direct coordinate products at a fixed primal quartet mask.

Coordinates/steps are in Bohr and arrays use float64/complex128. The large
cutoff deliberately retains only part of the quartet set; these are algebra
checks, not physical derivative-error bounds. Dense ERIs belong only to the
small independent oracle, prepared before the direct-path materialization ban.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.plan import IntegralPlan
from test_direct_density_screen import CUTOFF, mask_for, system


CASES = [
    pytest.param((True, True), 0., id="jk-coulomb"),
    pytest.param((True, False), 0., id="j-coulomb"),
    pytest.param((False, True), .3, id="k-long-range"),
    pytest.param((True, True), .3, id="jk-long-range"),
]
STEPS = (1e-4, 3e-5)


def _problem(system, selection, omega):
    parameters, plan = system.parameters, system.plan
    nnuc = len(system.top.nuclear_charges)
    coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    u = .07*jnp.sin(jnp.arange(coordinates.size).reshape(coordinates.shape)+.2)
    v = .09*jnp.cos(jnp.arange(coordinates.size).reshape(coordinates.shape)+.4)
    with_j, with_k = selection
    mask = mask_for(system.eri[omega], system.screen, system.offsets,
                    with_j=with_j, with_k=with_k)
    assert mask.any() and not mask.all(), "Exercise a partial primal quartet mask"
    frozen_mask = jnp.asarray(mask)

    def at(coords):
        return replace(parameters, nuclear_coords=coords[:nnuc], centers=coords[nnuc:])

    def oracle(coords, density):
        eri = plan.evaluate("eri", at(coords), omega=omega)*frozen_mask
        return (jnp.einsum("pqrs,...rs->...pq", eri, density) if with_j else jnp.zeros_like(density),
                jnp.einsum("prqs,...rs->...pq", eri, density) if with_k else jnp.zeros_like(density))

    def direct(coords, density):
        return plan.get_jk(at(coords), density, screen_density=system.screen,
                           screening_threshold=CUTOFF, workers=2, omega=omega,
                           with_j=with_j, with_k=with_k)

    return coordinates, u, v, at, mask, oracle, direct


def _assert_stable_mask(system, at, coordinates, direction, mask, selection, omega):
    for step in STEPS:
        for sign in (-1, 1):
            eri = np.asarray(system.plan.evaluate("eri", at(coordinates+sign*step*direction), omega=omega))
            displaced = mask_for(eri, system.screen, system.offsets,
                                 with_j=selection[0], with_k=selection[1])
            np.testing.assert_array_equal(displaced, mask,
                                          err_msg=f"Mask crossed a threshold at step {sign*step} Bohr")


def _forbid_eri(monkeypatch):
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("Screened direct geometry must not construct AO or packed ERIs")
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", checked)


def _compare(actual, expected, *, atol=3e-9, rtol=3e-9):
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(observed)).all()
        np.testing.assert_allclose(observed, reference, atol=atol, rtol=rtol)


def _weights(density):
    rng = np.random.default_rng(20261110)
    return tuple(jnp.asarray(rng.normal(size=density.shape)+1j*rng.normal(size=density.shape))
                 for _ in range(2))


def _loss(values, weights):
    return sum(jnp.vdot(weight, value).real for value, weight in zip(values, weights))


@pytest.mark.parametrize("selection,omega", CASES)
def test_screened_geometry_first_products_match_frozen_mask(system, monkeypatch, selection, omega):
    r, u, _, at, mask, oracle, direct = _problem(system, selection, omega)
    density = system.density  # nonsymmetric complex payload, reference batches differ
    weights = _weights(density)
    expected = jax.jit(lambda q: jax.jvp(lambda x: oracle(x, density), (q,), (u,)))(r)
    expected_gradient = jax.jit(jax.grad(lambda q: _loss(oracle(q, density), weights)))(r)
    _assert_stable_mask(system, at, r, u, mask, selection, omega)
    _forbid_eri(monkeypatch)

    observed = jax.jit(lambda q: jax.jvp(lambda x: direct(x, density), (q,), (u,)))(r)
    gradient = jax.jit(jax.grad(lambda q: _loss(direct(q, density), weights)))(r)
    _compare(observed, expected)
    _compare(gradient, expected_gradient)
    np.testing.assert_allclose(_loss(observed[1], weights), jnp.vdot(gradient, u), atol=3e-9, rtol=3e-9)
    for step in STEPS:
        fd = jax.tree.map(lambda a, b: (a-b)/(2*step), direct(r+step*u, density), direct(r-step*u, density))
        _compare(observed[1], fd, atol=3e-8, rtol=3e-6)


@pytest.mark.parametrize("selection,omega", CASES)
def test_screened_geometry_second_products_match_frozen_mask(system, monkeypatch, selection, omega):
    r, u, v, at, mask, oracle, direct = _problem(system, selection, omega)
    density, weights = system.density, _weights(system.density)

    def second(action, coords, left, right):
        first = lambda q: jax.jvp(lambda x: action(x, density), (q,), (left,))[1]
        return jax.jvp(first, (coords,), (right,))[1]

    expected_second = jax.jit(lambda q: second(oracle, q, u, v))(r)
    reference_gradient = jax.grad(lambda q: _loss(oracle(q, density), weights))
    expected_hvp = jax.jit(lambda q: jax.jvp(reference_gradient, (q,), (v,))[1])(r)
    _assert_stable_mask(system, at, r, v, mask, selection, omega)
    _forbid_eri(monkeypatch)

    observed_second = jax.jit(lambda q: second(direct, q, u, v))(r)
    swapped = jax.jit(lambda q: second(direct, q, v, u))(r)
    gradient = jax.jit(jax.grad(lambda q: _loss(direct(q, density), weights)))
    observed_hvp = jax.jit(lambda q: jax.jvp(gradient, (q,), (v,))[1])(r)
    _compare(observed_second, expected_second)
    _compare(observed_second, swapped)
    _compare(observed_hvp, expected_hvp)
    np.testing.assert_allclose(_loss(observed_second, weights), jnp.vdot(observed_hvp, u),
                               atol=3e-9, rtol=3e-9)
    for step in STEPS:
        fd = (gradient(r+step*v)-gradient(r-step*v))/(2*step)
        _compare(observed_hvp, fd, atol=5e-8, rtol=4e-6)


@pytest.mark.parametrize("selection,omega", CASES)
def test_screened_geometry_density_mixed_response_at_zero_payload(system, monkeypatch, selection, omega):
    r, u, _, at, mask, oracle, direct = _problem(system, selection, omega)
    zero = jnp.zeros_like(system.density)
    direction = .13*jnp.sin(system.density)
    weights = _weights(zero)
    expected = jax.jit(lambda q: jax.jvp(lambda x: oracle(x, direction), (q,), (u,))[1])(r)
    reference_gradient = jax.grad(lambda q, d: _loss(oracle(q, d), weights), (0, 1))
    expected_hvp = jax.jit(lambda q, d: jax.jvp(reference_gradient, (q, d), (u, direction))[1])(r, zero)
    assert sum(float(jnp.linalg.norm(x)) for x in expected) > 1e-8
    _assert_stable_mask(system, at, r, u, mask, selection, omega)
    _forbid_eri(monkeypatch)

    def mixed(coords, density):
        first = lambda q: jax.jvp(lambda d: direct(q, d), (density,), (direction,))[1]
        return jax.jvp(first, (coords,), (u,))[1]

    observed = jax.jit(mixed)(r, zero)
    gradient = jax.jit(jax.grad(lambda q, d: _loss(direct(q, d), weights), (0, 1)))
    observed_hvp = jax.jit(lambda q, d: jax.jvp(gradient, (q, d), (u, direction))[1])(r, zero)
    _compare(direct(r, zero), (zero, zero), atol=0., rtol=0.)
    _compare(observed, expected)
    _compare(observed_hvp, expected_hvp)
    for step in STEPS:
        pp = direct(r+step*u, step*direction)
        pm = direct(r+step*u, -step*direction)
        mp = direct(r-step*u, step*direction)
        mm = direct(r-step*u, -step*direction)
        fd = jax.tree.map(lambda a, b, c, d: (a-b-c+d)/(4*step**2), pp, pm, mp, mm)
        _compare(observed, fd, atol=5e-8, rtol=4e-6)
        plus, minus = gradient(r+step*u, step*direction), gradient(r-step*u, -step*direction)
        fd_hvp = jax.tree.map(lambda a, b: (a-b)/(2*step), plus, minus)
        _compare(observed_hvp, fd_hvp, atol=5e-8, rtol=4e-6)
