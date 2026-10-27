"""Public packed J/K selection, bounded execution and independent AD oracles."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.molecular.jk import build_jk_from_packed


def _problem(layout, n=5):
    rng = np.random.default_rng(2781)
    rows, cols = np.tril_indices(n)
    pair = np.empty((n, n), dtype=np.int32)
    pair[rows, cols] = pair[cols, rows] = np.arange(len(rows))
    values = rng.normal(size=(len(rows), len(rows)))
    values = .5 * (values + values.T)
    eri = values if layout == "s4" else values[np.tril_indices(len(rows))]
    density = rng.normal(size=(2, 3, n, n)) + 1j*rng.normal(size=(2, 3, n, n))
    p, q = pair[:, :, None, None], pair[None, None, :, :]
    if layout == "s8":
        p, q = np.maximum(p, q), np.minimum(p, q)
        indices = (p*(p+1)//2+q,)
    else:
        indices = (p, q)

    def oracle(e, d, with_j=True, with_k=True):
        full = e[indices]
        return (jnp.einsum("pqrs,...rs->...pq", full, d) if with_j else jnp.zeros_like(d),
                jnp.einsum("prqs,...rs->...pq", full, d) if with_k else jnp.zeros_like(d))

    return jnp.asarray(eri), jnp.asarray(density), oracle


@pytest.fixture(params=["cpu", "gpu"])
def execution_device(request):
    try:
        devices = jax.devices(request.param)
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip(f"{request.param} backend is unavailable")
    return devices[0]


@pytest.mark.parametrize("layout", ["s4", "s8"])
@pytest.mark.parametrize("with_j,with_k", [(True, True), (True, False), (False, True), (False, False)])
def test_selected_packed_value_jvp_vjp(layout, with_j, with_k, execution_device):
    eri, density, oracle = _problem(layout)
    eri, density = (jax.device_put(x, execution_device) for x in (eri, density))
    action = lambda e, d: build_jk_from_packed(
        e, d, with_j=with_j, with_k=with_k, max_memory=1.)
    reference = lambda e, d: oracle(e, d, with_j, with_k)
    directions = (.03*jnp.cos(eri), .07*jnp.sin(density))
    actual = jax.jit(lambda e, d, de, dd: jax.jvp(action, (e, d), (de, dd)))(
        eri, density, *directions)
    expected = jax.jvp(reference, (eri, density), directions)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=3e-11, rtol=3e-11)
        assert a.device == execution_device
    loss = lambda fn, e, d: sum(jnp.sum(jnp.abs(x)**2) for x in fn(e, d))
    actual_grad = jax.jit(jax.grad(lambda e, d: loss(action, e, d), (0, 1)))(eri, density)
    expected_grad = jax.grad(lambda e, d: loss(reference, e, d), (0, 1))(eri, density)
    for a, b in zip(actual_grad, expected_grad):
        np.testing.assert_allclose(a, b, atol=3e-9, rtol=3e-11)


@pytest.mark.parametrize("layout", ["s4", "s8"])
def test_forced_bounded_rows_handle_remainder_and_higher_density_ad(layout):
    eri, density, oracle = _problem(layout)
    action = lambda d: build_jk_from_packed(eri, d, block_size=2, max_memory=.1)
    for actual, expected in zip(jax.jit(action)(density), oracle(eri, density)):
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)
    loss = lambda d: sum(jnp.sum(jnp.abs(x)**2) for x in action(d))
    reference = lambda d: sum(jnp.sum(jnp.abs(x)**2) for x in oracle(eri, d))
    direction = .1*jnp.cos(density)
    actual = jax.jit(lambda d, v: jax.jvp(jax.grad(loss), (d,), (v,))[1])(density, direction)
    expected = jax.jvp(jax.grad(reference), (density,), (direction,))[1]
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-11)


def _aval_shapes(jaxpr):
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
        for value in jaxpr.values():
            yield from _aval_shapes(value)


def test_bounded_rows_and_eri_vjp_never_declare_full_eri():
    n = 9
    eri, density, _ = _problem("s4", n)
    action = lambda e: build_jk_from_packed(e, density, block_size=2, max_memory=.1)
    loss = lambda e: sum(jnp.sum(jnp.abs(x)**2) for x in action(e))
    for function in (action, jax.grad(loss)):
        shapes = tuple(_aval_shapes(jax.make_jaxpr(function)(eri)))
        assert (n, n, n, n) not in shapes
        assert (2, n, n, n) in shapes
        for shape in shapes:
            if any(shape[i:i+3] == (n, n, n) for i in range(len(shape)-2)):
                # A gather may carry a final two-component index axis. A
                # reverse scan must not stack those indices for all tiles.
                assert np.prod(shape) <= 2*2*n**3, shape


@pytest.mark.parametrize("layout", ["s4", "s8"])
def test_tiny_workspace_subdivides_exchange_rows(layout):
    eri, density, oracle = _problem(layout)
    density = density[0, 0]
    action = lambda e, d: build_jk_from_packed(e, d, block_size=2, max_memory=.005)
    for actual, expected in zip(jax.jit(action)(eri, density), oracle(eri, density)):
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)


def test_disabled_operators_ignore_contraction_tile_workspace():
    eri,density,_=_problem('s4')
    for block in (None,1):
        j,k=build_jk_from_packed(eri,density,with_j=False,with_k=False,
                               block_size=block,max_memory=1e-6)
        np.testing.assert_array_equal(j,jnp.zeros_like(density))
        np.testing.assert_array_equal(k,jnp.zeros_like(density))


def test_native_s8_eri_transpose_preserves_selection_and_workspace(monkeypatch):
    from gradscf.integrals.backends.native import jk as native
    from gradscf.integrals.molecular import jk as molecular
    observed=[]
    original=molecular._build_jk_from_packed_jax
    def record(e,d,**options):
        observed.append(options)
        return original(e,d,**options)
    monkeypatch.setattr(molecular,'_build_jk_from_packed_jax',record)
    native._packed_operator.cache_clear()
    eri,density,_=_problem('s8')
    loss=lambda e:jnp.sum(native.packed_jk(e,density,with_k=False,max_memory=.03)[0].real)
    jax.make_jaxpr(jax.grad(loss))(eri)
    assert observed
    assert all(x.get('with_j') is True and x.get('with_k') is False
               and x.get('max_memory') == .03 for x in observed)


@pytest.mark.parametrize("options", [{"block_size": 0}, {"max_memory": 0.},
                                    {"max_memory": float("nan")}, {"max_memory": -1.}])
def test_invalid_workspace_options_fail_explicitly(options):
    eri, density, _ = _problem("s4")
    with pytest.raises(ValueError, match="(?i)(block_size|max_memory|workspace)"):
        build_jk_from_packed(eri, density, **options)


@pytest.mark.parametrize("layout", ["s4", "s8"])
def test_selected_packed_jk_preserves_geometry_products(layout):
    from gradscf import integrals

    top, parameters = integrals.prepare_basis("H 0 0 0; H .2 -.1 1.2", "sto-3g")
    plan = integrals.make_plan(top)
    coords = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    density = jnp.asarray([[.8, .2j], [.1, .4]], dtype=jnp.complex128)
    direction = .1*jnp.sin(jnp.arange(coords.size).reshape(coords.shape))

    def action(r, packed):
        p = replace(parameters, nuclear_coords=r[:2], centers=r[2:])
        eri = plan.evaluate("eri", p, aosym=layout if packed else "s1")
        if packed:
            return build_jk_from_packed(eri, density, with_j=False, max_memory=1.)[1]
        return jnp.einsum("prqs,rs->pq", eri, density)

    loss = lambda r, packed: jnp.sum(jnp.abs(action(r, packed))**2)
    evaluate = lambda r, packed: (jax.jvp(lambda x: action(x, packed), (r,), (direction,)),
        jax.jvp(jax.grad(lambda x: loss(x, packed)), (r,), (direction,)))
    actual = jax.jit(lambda r: evaluate(r, True))(coords)
    expected = evaluate(coords, False)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=3e-10, rtol=3e-10)
