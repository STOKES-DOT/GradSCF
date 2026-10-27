"""Packed ERI coordinate products stay packed (CPU float64; coordinates in Bohr)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


def _problem(cart, layout, omega=0.):
    # General contractions and a d shell distinguish Cartesian/spherical order.
    top, params = integrals.prepare_basis(
        "H 0 0 0; He .2 -.1 .8",
        {"H": [[0, [1.1, .8, -.2], [.3, .2, .9]], [2, [.7, 1.]]],
         "He": [[0, [.8, 1.]], [1, [.5, 1.]]]}, cart=cart)
    plan = integrals.make_plan(top)
    coords = jnp.concatenate((params.nuclear_coords, params.centers))

    def value(r, aosym=layout):
        return plan.evaluate("eri", replace(params, nuclear_coords=r[:2], centers=r[2:]),
                             aosym=aosym, omega=omega)

    rows, cols = np.tril_indices(top.nao)

    def reference(r):
        full = value(r, "s1")
        pair = full[rows[:, None], cols[:, None], rows[None, :], cols[None, :]]
        return pair if layout == "s4" else pair[np.tril_indices(len(rows))]

    return value, reference, coords, top.nao


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("layout", ["s4", "s8"])
def test_packed_coordinate_jvp_vjp_and_second_products(cart, layout):
    value, reference, coords, _ = _problem(cart, layout)
    rng = np.random.default_rng(731)
    u, v = [jnp.asarray(rng.normal(size=coords.shape)) * .1 for _ in range(2)]
    actual, tangent = jax.jit(lambda r, d: jax.jvp(value, (r,), (d,)))(coords, v)
    expected, expected_tangent = jax.jvp(reference, (coords,), (v,))
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=1e-11)
    np.testing.assert_allclose(tangent, expected_tangent, atol=3e-11, rtol=2e-10)
    cotangent = jnp.asarray(rng.normal(size=actual.shape))
    pull = lambda r: jax.grad(lambda x: jnp.vdot(value(x), cotangent))(r)
    expected_pull = lambda r: jax.grad(lambda x: jnp.vdot(reference(x), cotangent))(r)
    gradient = jax.jit(pull)(coords)
    np.testing.assert_allclose(gradient, expected_pull(coords), atol=3e-10, rtol=2e-10)
    np.testing.assert_allclose(jnp.vdot(gradient, v), jnp.vdot(cotangent, tangent), atol=3e-10)
    first = lambda r: jax.jvp(value, (r,), (v,))[1]
    second = jax.jit(lambda r: jax.jvp(first, (r,), (u,))[1])(coords)
    expected_second = jax.jvp(lambda r: jax.jvp(reference, (r,), (v,))[1],
                              (coords,), (u,))[1]
    np.testing.assert_allclose(second, expected_second, atol=4e-11, rtol=3e-10)
    hvp = jax.jit(lambda r: jax.jvp(pull, (r,), (v,))[1])(coords)
    np.testing.assert_allclose(hvp, jax.jvp(expected_pull, (coords,), (v,))[1],
                               atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(jnp.vdot(hvp, u), jnp.vdot(cotangent, second), atol=3e-10)
    h = 2e-4
    finite_difference = (first(coords+h*u)-first(coords-h*u))/(2*h)
    np.testing.assert_allclose(second, finite_difference, atol=3e-8, rtol=3e-7)


@pytest.mark.parametrize("layout", ["s4", "s8"])
def test_packed_coordinate_products_transpose_directions_and_batch(layout):
    value, _, coords, _ = _problem(False, layout)
    rng = np.random.default_rng(32)
    u, v = [jnp.asarray(rng.normal(size=coords.shape)) * .1 for _ in range(2)]
    cotangent = jnp.asarray(rng.normal(size=value(coords).shape))
    second = lambda a, b: jax.jvp(
        lambda r: jax.jvp(value, (r,), (b,))[1], (coords,), (a,))[1]
    expected = jax.jvp(jax.grad(lambda r: jnp.vdot(value(r), cotangent)),
                       (coords,), (v,))[1]
    transposed = jax.jit(jax.grad(lambda a: jnp.vdot(second(a, v), cotangent)))(u)
    np.testing.assert_allclose(transposed, expected, atol=3e-10, rtol=3e-10)
    pull = lambda c: jax.jvp(jax.grad(lambda r: jnp.vdot(value(r), c)),
                            (coords,), (v,))[1]
    np.testing.assert_allclose(jax.grad(lambda c: jnp.vdot(pull(c), u))(cotangent),
                               second(u, v), atol=3e-10, rtol=3e-10)
    batched = jax.jit(jax.vmap(lambda d: second(d, v)))(jnp.stack((u, 2*u)))
    np.testing.assert_allclose(batched, jnp.stack((second(u, v), 2*second(u, v))),
                               atol=3e-11, rtol=3e-10)


def _aval_shapes(jaxpr):
    """Inspect nested primitive outputs, including custom-JVP primal jaxprs."""
    if hasattr(jaxpr, "jaxpr"):
        yield from _aval_shapes(jaxpr.jaxpr)
    elif hasattr(jaxpr, "eqns"):
        for equation in jaxpr.eqns:
            for variable in equation.outvars:
                yield getattr(variable.aval, "shape", ())
            for parameter in equation.params.values():
                yield from _aval_shapes(parameter)
    elif isinstance(jaxpr, (tuple, list)):
        for item in jaxpr:
            yield from _aval_shapes(item)
    elif isinstance(jaxpr, dict):
        for item in jaxpr.values():
            yield from _aval_shapes(item)


@pytest.mark.parametrize("layout", ["s4", "s8"])
def test_water_packed_coordinate_products_never_declare_full_eri(layout):
    top, params = integrals.prepare_basis(
        "O 0 0 0; H 0 .75 .58; H 0 -.75 .58", "6-31g(d,p)", cart=True)
    assert top.nao == 25
    plan = integrals.make_plan(top)
    coords = jnp.concatenate((params.nuclear_coords, params.centers))
    value = lambda r: plan.evaluate("eri", replace(params, nuclear_coords=r[:3], centers=r[3:]),
                                    aosym=layout)
    first = lambda r, d: jax.jvp(value, (r,), (d,))[1]
    second = lambda r, u, v: jax.jvp(lambda x: first(x, v), (r,), (u,))[1]
    n = top.nao
    pairs = n*(n+1)//2
    shape = (pairs, pairs) if layout == "s4" else (pairs*(pairs+1)//2,)
    assert jax.eval_shape(value, coords).shape == shape
    for function, args in ((first, (coords, coords)), (second, (coords, coords, coords))):
        shapes = tuple(_aval_shapes(jax.make_jaxpr(function)(*args)))
        assert (n, n, n, n) not in shapes
        assert shape in shapes


def test_packed_third_coordinate_derivative_fails_explicitly():
    value, _, coords, _ = _problem(True, "s8")
    first = lambda r: jax.jvp(value, (r,), (jnp.ones_like(r),))[1]
    second = lambda r: jax.jvp(first, (r,), (jnp.ones_like(r),))[1]
    with pytest.raises(NotImplementedError, match="(?i)(packed|compact|third)"):
        jax.jvp(second, (coords,), (jnp.ones_like(coords),))


@pytest.mark.parametrize("cart", [True, False])
@pytest.mark.parametrize("layout", ["s4", "s8"])
def test_packed_long_range_value_and_second_coordinate_product(cart, layout):
    value, reference, coords, _ = _problem(cart, layout, omega=.2)
    direction = jnp.sin(jnp.arange(coords.size).reshape(coords.shape))*.1
    np.testing.assert_allclose(value(coords), reference(coords), atol=2e-11, rtol=1e-11)
    second = lambda f, r: jax.jvp(lambda x: jax.jvp(f, (x,), (direction,))[1],
                                  (r,), (direction,))[1]
    np.testing.assert_allclose(jax.jit(lambda r: second(value, r))(coords), second(reference, coords),
                               atol=3e-11, rtol=3e-10)


def test_raw_packed_long_range_value_matches_full_gather():
    from gradscf.integrals.backends.native.ffi import evaluate_compact

    top, params = integrals.prepare_basis("H 0 0 0; H .2 .1 .8", "3-21g")
    plan = integrals.make_plan(top)
    atm, bas, env = plan.pack(params)
    pairs = top.nao*(top.nao+1)//2
    raw = evaluate_compact(atm, bas, env.at[8].set(.2), (pairs, pairs),
                            layout=0, cart=top.cart)
    full = plan.evaluate("eri", params, omega=.2)
    rows, cols = np.tril_indices(top.nao)
    expected = full[rows[:, None], cols[:, None], rows[None, :], cols[None, :]]
    np.testing.assert_allclose(raw, expected, atol=2e-11, rtol=1e-11)


def _thread_products(layout="s8"):
    top, params = integrals.prepare_basis("H 0 0 0; H .2 .1 .8", "3-21g")
    plan = integrals.make_plan(top)
    coords = jnp.concatenate((params.nuclear_coords, params.centers))
    def loss(r):
        packed = plan.evaluate("eri", replace(params, nuclear_coords=r[:2], centers=r[2:]),
                                aosym=layout)
        return jnp.vdot(packed, jnp.cos(jnp.arange(packed.size)))
    gradient = jax.jit(jax.grad(loss))
    direction = jnp.sin(jnp.arange(coords.size).reshape(coords.shape))*.1
    hessian = jax.jit(lambda r: jax.jvp(gradient, (r,), (direction,))[1])
    return gradient, hessian, coords


@pytest.mark.parametrize("order", [1, 2])
def test_packed_coordinate_adjoint_rejects_invalid_thread_limit(monkeypatch, order):
    gradient, hessian, coords = _thread_products()
    monkeypatch.setenv("GRADSCF_NATIVE_THREADS", "0")
    with pytest.raises(Exception, match="GRADSCF_NATIVE_THREADS"):
        jax.block_until_ready((gradient if order == 1 else hessian)(coords))


@pytest.mark.parametrize("layout", ["s4", "s8"])
def test_packed_coordinate_adjoint_cpu_threads_match_serial(monkeypatch, layout):
    gradient, hessian, coords = _thread_products(layout)
    monkeypatch.setenv("GRADSCF_NATIVE_THREADS", "1")
    serial = [jax.block_until_ready(f(coords)) for f in (gradient, hessian)]
    monkeypatch.setenv("GRADSCF_NATIVE_THREADS", "2")
    for function, expected in zip((gradient, hessian), serial):
        actual = jax.block_until_ready(function(coords))
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-10)
        np.testing.assert_array_equal(jax.block_until_ready(function(coords)), actual)
