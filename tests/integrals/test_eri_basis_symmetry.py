"""S9 four-center symmetry oracles; CPU float64, Bohr and atomic units.

Values/FD are independent of the derivative scheduler. Diagnostic call-count
nodes require a separate instrumented native build; ordinary builds skip them.
"""
from dataclasses import replace
from itertools import product
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.autodiff.eri_basis import evaluate_eri_basis


WINDOW = (0, 1, 1, 2, 0, 2, 0, 2)


def _problem(cart=True, *, single=False, four=False):
    shells = ([[0, [.8, 1.]]] if single else
              [[0, [1.3, .7, .2], [.45, 0., .6], [.18, .2, -.1]],
               [2, [.9, .6], [.25, .4]]])
    atom, basis = "He 0 0 0", {"He": shells}
    if four:
        atom = "He 0 0 0; H .2 .5 .7; Li -.6 .3 .2; Be .4 -.5 .8"
        basis = {
            "He": [[1, [1.3, .7, .2], [.45, 0., .6], [.18, .2, -.1]]],
            "H": [[0, [.9, .6], [.25, .4]]],
            "Li": [[0, [.6, .8], [.2, .3]]],
            "Be": [[2, [.7, .5], [.21, .2]]],
        }
    top, p = integrals.prepare_basis(atom, basis, unit="Bohr", cart=cart)
    plan = integrals.make_plan(top)
    fixed = replace(p, centers=jnp.zeros_like(p.centers), nuclear_coords=jnp.zeros_like(p.nuclear_coords))
    atm, bas, env, c, a = plan.coefficient_data(fixed, separate_exponents=True)
    coords = jnp.concatenate((p.nuclear_coords, p.centers))
    def value(coefficients, exponents=a, bounds=None):
        return evaluate_eri_basis(atm, bas, env, coefficients, coords, jnp.zeros(3),
            top.nao, cart=cart, exponents=exponents, shell_slice=bounds)
    return top, p, value, c, a


def _second(value, c, u, v):
    return jax.jvp(lambda x: jax.jvp(value, (x,), (v,))[1], (c,), (u,))[1]


def test_single_shell_diagonal_keeps_four_and_twelve_derivative_occurrences():
    _, _, value, _, _ = _problem(single=True)
    c, u, v = [jnp.array([x]) for x in (.6, .11, -.07)]
    unit = value(jnp.ones(1))
    tangent = jax.jit(lambda x: jax.jvp(value, (x,), (u,))[1])(c)
    bilinear = jax.jit(lambda x: _second(value, x, u, v))(c)
    np.testing.assert_allclose(value(c), c[0]**4*unit, atol=2e-12)
    np.testing.assert_allclose(tangent, 4*c[0]**3*u[0]*unit, atol=2e-12)
    np.testing.assert_allclose(bilinear, 12*c[0]**2*u[0]*v[0]*unit, atol=2e-12)
    cot = jnp.full((1,)*4, .37)
    gradient = jax.grad(lambda x: jnp.vdot(value(x), cot))
    hvp = jax.jvp(gradient, (c,), (u,))[1]
    np.testing.assert_allclose(jnp.vdot(gradient(c), u), jnp.vdot(tangent, cot), atol=2e-12)
    np.testing.assert_allclose(jnp.vdot(hvp, v), jnp.vdot(bilinear, cot), atol=2e-12)


@pytest.mark.parametrize("cart", [True, False])
def test_asymmetric_overlapping_window_pools_all_nonsymmetric_cotangents(cart):
    top, _, value, c, a = _problem(cart)
    # Every canonical shell tuple in this window has its first shell OUTSIDE
    # window axis zero. Last-axis overlap admits multiple symmetry copies.
    ns, nd = 2, 6 if cart else 5
    view = (slice(0, ns), slice(ns, ns+nd), slice(None), slice(None))
    full = lambda x, y: value(x, y)
    window = lambda x, y: value(x, y, WINDOW)
    u = jnp.sin(jnp.arange(c.size)+.7)*.03
    v = jnp.cos(jnp.arange(c.size)+.4)*.04
    da = jnp.sin(jnp.arange(a.size)+.3)*.02
    actual, tangent = jax.jit(lambda x, y: jax.jvp(window, (x, y), (u, da)))(c, a)
    expected, expected_tangent = jax.jvp(full, (c, a), (u, da))
    assert actual.shape == (ns, nd, top.nao, top.nao)
    np.testing.assert_allclose(actual, expected[view], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(tangent, expected_tangent[view], atol=3e-10, rtol=3e-9)
    cot = jnp.asarray(np.random.default_rng(141).normal(size=actual.shape))*.01
    cot = cot.at[0, 0, 0, 0].add(1.)
    embedded = jnp.zeros((top.nao,)*4).at[view].set(cot)
    _, pull = jax.vjp(window, c, a)
    _, reference_pull = jax.vjp(full, c, a)
    gc, ga = pull(cot)
    rc, ra = reference_pull(embedded)
    np.testing.assert_allclose(gc, rc, atol=3e-10, rtol=3e-9)
    np.testing.assert_allclose(ga, ra, atol=3e-10, rtol=3e-9)
    np.testing.assert_allclose(jnp.vdot(cot, tangent), jnp.vdot(gc, u)+jnp.vdot(ga, da), atol=3e-10)
    first_c = lambda x: window(x, a)
    second = jax.jit(lambda x: _second(first_c, x, u, v))(c)
    reference_second = _second(lambda x: full(x, a), c, u, v)[view]
    np.testing.assert_allclose(second, reference_second, atol=3e-10, rtol=3e-9)
    gradient = jax.jit(jax.grad(lambda x: jnp.vdot(window(x, a), cot)))
    hvp = jax.jvp(gradient, (c,), (u,))[1]
    np.testing.assert_allclose(jnp.vdot(hvp, v), jnp.vdot(second, cot), atol=3e-10)
    for h in (2e-4, 1e-4):
        fd = (window(c+h*u, a+h*da)-window(c-h*u, a-h*da))/(2*h)
        np.testing.assert_allclose(tangent, fd, atol=3e-8, rtol=3e-6)
        fd_hvp = (gradient(c+h*u)-gradient(c-h*u))/(2*h)
        np.testing.assert_allclose(hvp, fd_hvp, atol=2e-7, rtol=4e-6)



@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("bounds", [None, (0, 2, 1, 4, 0, 3, 2, 4)], ids=["full", "window"])
def test_four_center_basis_products_preserve_thread_ownership(monkeypatch, cart, bounds):
    _, _, value, c, a = _problem(cart, four=True)
    function = lambda x, y: value(x, y, bounds)
    u = jnp.sin(jnp.arange(c.size)+.7)*.03
    v = jnp.cos(jnp.arange(c.size)+.4)*.04
    da = jnp.sin(jnp.arange(a.size)+.3)*.02
    shape = function(c, a).shape
    cot = jnp.asarray(np.random.default_rng(321).normal(size=shape))*.01
    gradient = jax.grad(lambda x, y: jnp.vdot(function(x, y), cot), argnums=(0, 1))

    @jax.jit
    def products(x, y):
        coefficients = lambda z: function(z, y)
        tc = jax.jvp(coefficients, (x,), (u,))[1]
        ta = jax.jvp(lambda z: function(x, z), (y,), (da,))[1]
        gc, ga = gradient(x, y)
        second = _second(coefficients, x, u, v)
        coefficient_gradient = jax.grad(lambda z: jnp.vdot(function(z, y), cot))
        hvp = jax.jvp(coefficient_gradient, (x,), (u,))[1]
        return tc, ta, gc, ga, second, hvp

    monkeypatch.setenv("GRADSCF_NATIVE_THREADS", "1")
    serial = jax.block_until_ready(products(c, a))
    tc, ta, gc, ga, second, hvp = serial
    np.testing.assert_allclose(jnp.vdot(cot, tc+ta), jnp.vdot(gc, u)+jnp.vdot(ga, da),
                               atol=3e-10, rtol=3e-9)
    np.testing.assert_allclose(jnp.vdot(hvp, v), jnp.vdot(second, cot), atol=3e-10, rtol=3e-9)
    h = 1e-4
    fd = (function(c+h*u, a+h*da)-function(c-h*u, a-h*da))/(2*h)
    np.testing.assert_allclose(tc+ta, fd, atol=3e-8, rtol=3e-6)
    fd_hvp = (gradient(c+h*u, a)[0]-gradient(c-h*u, a)[0])/(2*h)
    np.testing.assert_allclose(hvp, fd_hvp, atol=2e-7, rtol=4e-6)
    for threads in (2, 4, 32):
        monkeypatch.setenv("GRADSCF_NATIVE_THREADS", str(threads))
        actual = jax.block_until_ready(products(c, a))
        for observed, expected in zip(actual, serial):
            np.testing.assert_allclose(observed, expected, atol=3e-10, rtol=3e-9)


def _canonical_shell_tuple(quartet):
    left, right = [tuple(sorted(quartet[start:start+2], reverse=True)) for start in (0, 2)]
    first, second = sorted((left, right), reverse=True)
    return first+second


def _independent_shells(nshell, bounds):
    ranges = (tuple(range(nshell)) for _ in range(4)) if bounds is None else (
        tuple(range(start, stop)) for start, stop in zip(bounds[::2], bounds[1::2]))
    return {_canonical_shell_tuple(q) for q in product(*ranges)}


@pytest.fixture
def basis_counters():
    required = os.environ.get("GRADSCF_EXPECT_DIAGNOSTICS") == "1"
    try:
        from gradscf.integrals._native.diagnostics import get_counts, reset_counts
    except ImportError:
        if required:
            pytest.fail("S9 call-count acceptance requires the diagnostic native build")
        pytest.skip("S9 counters are provided only by the separate diagnostic native build")
    if not get_counts()["enabled"]:
        if required:
            pytest.fail("S9 call-count acceptance requires enabled diagnostic counters")
        pytest.skip("S9 counters are disabled in the ordinary production native build")
    return reset_counts, get_counts


@pytest.mark.parametrize("bounds", [None, WINDOW], ids=["full", "window"])
@pytest.mark.parametrize("kind", ["coefficient_jvp", "coefficient_vjp", "coefficient_hessian_jvp",
                                  "coefficient_hessian_vjp", "exponent_jvp", "exponent_vjp"])
def test_diagnostic_basis_kernel_calls_use_independent_shell_orbits(monkeypatch, basis_counters, bounds, kind):
    monkeypatch.setenv("GRADSCF_NATIVE_THREADS", "1")
    top, _, value, c, a = _problem()
    function = lambda x, y: value(x, y, bounds)
    u = jnp.sin(jnp.arange(c.size)+.7)*.03
    v = jnp.cos(jnp.arange(c.size)+.4)*.04
    da = jnp.sin(jnp.arange(a.size)+.3)*.02
    cot = jnp.cos(jnp.arange(function(c, a).size)).reshape(function(c, a).shape)
    if kind == "coefficient_jvp":
        call = jax.jit(lambda x: jax.jvp(lambda y: function(y, a), (x,), (u,))[1])
    elif kind == "coefficient_vjp":
        call = jax.jit(jax.grad(lambda x: jnp.vdot(function(x, a), cot)))
    elif kind == "coefficient_hessian_jvp":
        call = jax.jit(lambda x: _second(lambda y: function(y, a), x, u, v))
    elif kind == "coefficient_hessian_vjp":
        grad = jax.grad(lambda x: jnp.vdot(function(x, a), cot))
        call = jax.jit(lambda x: jax.jvp(grad, (x,), (u,))[1])
    elif kind == "exponent_jvp":
        call = jax.jit(lambda x: jax.jvp(lambda y: function(c, y), (x,), (da,))[1])
    else:
        call = jax.jit(jax.grad(lambda x: jnp.vdot(function(c, x), cot)))
    argument = a if kind.startswith("exponent") else c
    jax.block_until_ready(call(argument))
    reset, read = basis_counters
    reset()
    jax.block_until_ready(call(argument))
    counts = read()["families"]["eri4c"]
    visits, evaluations, queries = [counts[name] for name in
                                    ("scheduled_quartets", "evaluations", "cache_queries")]
    orbits = _independent_shells(len(top.angular_momenta), bounds)
    expected = (sum(sum(top.primitive_counts[s] for s in quartet) for quartet in orbits)
                if kind.startswith("exponent") else
                (12 if "hessian" in kind else 4)*len(orbits))
    assert visits == len(orbits), (kind, bounds, visits, len(orbits))
    assert 0 < evaluations <= expected, (kind, bounds, evaluations, expected)
    assert 0 <= queries <= evaluations, (kind, bounds, queries, evaluations)
