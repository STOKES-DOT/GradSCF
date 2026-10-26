"""S9 coordinate shell-orbit invariants; CPU float64, coordinates in Bohr.

The finite differences re-evaluate values. Nonsymmetric cotangents, unequal
directions and floating general contractions exercise derivative-position
permutations independently of the kernel-call-count acceptance gates.
"""
from dataclasses import replace
from itertools import product
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


WINDOW = (0, 1, 1, 2, 0, 2, 0, 2)
LEG_PERMUTATIONS = ((0, 1, 2, 3), (1, 0, 2, 3),
                    (2, 3, 0, 1), (3, 2, 0, 1))


def _problem(cart=True, layout="full", omega=0.):
    top, p = integrals.prepare_basis("He 0 0 0", {"He": [
        [0, [1.3, .7, .2], [.45, 0., .6]], [2, [.9, .6], [.25, .4]]]},
        unit="Bohr", cart=cart)
    plan = integrals.make_plan(top)
    coords = p.centers.at[1].set(jnp.array([.4, -.2, .7]))
    def value(r, selected=layout):
        return plan.evaluate("eri", replace(p, centers=r), omega=omega,
                             shell_slice=WINDOW if selected == "window" else None,
                             aosym=selected if selected in ("s4", "s8") else "s1")
    return top, value, coords


def _second(function, r, u, v):
    return jax.jvp(lambda x: jax.jvp(function, (x,), (v,))[1], (r,), (u,))[1]


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("layout", ["full", "window", "s4", "s8"])
def test_coordinate_orbits_preserve_adjoint_second_products_and_finite_difference(cart, layout):
    _, value, r = _problem(cart, layout)
    u = jnp.sin(jnp.arange(r.size).reshape(r.shape)+.2)*.13
    v = jnp.cos(jnp.arange(r.size).reshape(r.shape)+.4)*.09
    primal, first = jax.jit(lambda x: jax.jvp(value, (x,), (v,)))(r)
    cot = jnp.asarray(np.random.default_rng(439).normal(size=primal.shape))*.03
    gradient = jax.jit(jax.grad(lambda x: jnp.vdot(value(x), cot)))
    g = gradient(r)
    second = jax.jit(lambda x: _second(value, x, u, v))(r)
    hvp = jax.jit(lambda x: jax.jvp(gradient, (x,), (v,))[1])(r)
    np.testing.assert_allclose(jnp.vdot(first, cot), jnp.vdot(g, v), atol=3e-10, rtol=2e-10)
    np.testing.assert_allclose(jnp.vdot(second, cot), jnp.vdot(hvp, u), atol=3e-10, rtol=2e-10)
    np.testing.assert_allclose(second, _second(value, r, v, u), atol=3e-10, rtol=2e-10)
    for h in (2e-4, 1e-4):
        fd = (value(r+h*v)-value(r-h*v))/(2*h)
        np.testing.assert_allclose(first, fd, atol=3e-8, rtol=3e-6)
        fd_hvp = (gradient(r+h*v)-gradient(r-h*v))/(2*h)
        np.testing.assert_allclose(hvp, fd_hvp, atol=5e-8, rtol=4e-6)


def _canonical(quartet):
    pairs = [tuple(sorted(quartet[k:k+2], reverse=True)) for k in (0, 2)]
    a, b = sorted(pairs, reverse=True)
    return a+b


def _orbits(bounds):
    ranges = [range(2)]*4 if bounds is None else [range(a, b) for a, b in zip(bounds[::2], bounds[1::2])]
    return {_canonical(q) for q in product(*ranges)}


def _kernel_upper_bound(orbits, order):
    modes = [LEG_PERMUTATIONS]
    if order == 2:
        modes += [((0, 1, 2, 3), (2, 3, 0, 1)),
                  ((0, 1, 2, 3), (0, 1, 3, 2), (1, 0, 2, 3), (1, 0, 3, 2))]
    return sum(len({tuple(q[i] for i in p) for p in group}) for q in orbits for group in modes)


@pytest.fixture
def geometry_counters():
    required = os.environ.get("GRADSCF_EXPECT_DIAGNOSTICS") == "1"
    try:
        from gradscf.integrals._native.diagnostics import get_counts, reset_counts
    except ImportError:
        if required:
            pytest.fail("S9 coordinate call counts require the diagnostic native build")
        pytest.skip("Separate diagnostic native build is not installed")
    if not get_counts()["enabled"]:
        if required:
            pytest.fail("S9 coordinate call counts require enabled diagnostics")
        pytest.skip("Native counters are disabled in production builds")
    return reset_counts, get_counts


@pytest.mark.parametrize("layout", ["full", "window"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("transpose", [False, True], ids=["jvp", "vjp"])
def test_coordinate_kernel_calls_use_covariant_independent_orbits(monkeypatch, geometry_counters,
                                                                 layout, order, transpose):
    monkeypatch.setenv("GRADSCF_NATIVE_THREADS", "1")
    _, value, r = _problem(layout=layout)
    u = jnp.sin(jnp.arange(r.size).reshape(r.shape)+.2)*.13
    v = jnp.cos(jnp.arange(r.size).reshape(r.shape)+.4)*.09
    cot = jnp.cos(jnp.arange(value(r).size)).reshape(value(r).shape)
    if transpose:
        gradient = jax.grad(lambda x: jnp.vdot(value(x), cot))
        call = gradient if order == 1 else lambda x: jax.jvp(gradient, (x,), (u,))[1]
    else:
        call = (lambda x: jax.jvp(value, (x,), (u,))[1]) if order == 1 else (
            lambda x: _second(value, x, u, v))
    call = jax.jit(call)
    jax.block_until_ready(call(r))
    reset, read = geometry_counters
    reset()
    jax.block_until_ready(call(r))
    counts = read()["families"]["eri4c"]
    orbits = _orbits(WINDOW if layout == "window" else None)
    # The derivative pattern is part of the symmetry label: R1 has four leg
    # modes; R2 has four same-leg, two same-pair and four cross-pair modes.
    assert counts["scheduled_quartets"] == len(orbits)
    assert 0 < counts["evaluations"] <= _kernel_upper_bound(orbits, order)
