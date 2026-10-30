"""Matsubara outputs stay complete; factorization workspace is not frequency-batched."""
import jax
import jax.numpy as jnp
import numpy as np


def _lu_shapes(tree):
    if hasattr(tree, "jaxpr"):
        yield from _lu_shapes(tree.jaxpr)
    elif hasattr(tree, "eqns"):
        for equation in tree.eqns:
            if equation.primitive.name == "lu":
                yield equation.outvars[0].aval.shape
            yield from _lu_shapes(equation.params)
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _lu_shapes(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from _lu_shapes(value)


def test_screened_factorizations_are_frequency_bounded_and_differentiable():
    from gradscf.gw.matsubara import _screened
    rng = np.random.default_rng(911)
    pi = jnp.asarray(rng.normal(size=(9, 13, 13))*.01)
    expected = lambda x: jnp.linalg.solve(jnp.eye(13)-x, x)
    actual = jax.jit(_screened)(pi)
    np.testing.assert_allclose(actual, expected(pi), atol=1e-13)
    direction = jnp.ones_like(pi)*.002
    for function in (_screened, lambda x: jax.grad(lambda y: jnp.sum(_screened(y)**2))(x)):
        shapes = set(_lu_shapes(jax.make_jaxpr(function)(pi)))
        assert shapes and all(len(shape) == 2 for shape in shapes), shapes
    np.testing.assert_allclose(jax.jvp(_screened, (pi,), (direction,))[1],
        jax.jvp(expected, (pi,), (direction,))[1], atol=1e-13)
    np.testing.assert_allclose(jax.grad(lambda x: jnp.sum(_screened(x)**2))(pi),
        jax.grad(lambda x: jnp.sum(expected(x)**2))(pi), atol=1e-13)


def test_eager_charge_search_reuses_the_compiled_canonical_solver(monkeypatch):
    from gradscf.gw import scgw, scgw_response
    from gradscf.gw.matsubara import matsubara_grid
    grid = matsubara_grid(nw=8, beta=10.)
    fock = jnp.diag(jnp.array([-.4, .6]))
    sigma = jnp.zeros((16, 2, 2), dtype=jnp.complex128)
    calls, original = [], scgw_response._chemical_potential
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(scgw_response, "_chemical_potential", counted)
    chemical, green, density = scgw._solve_chemical_potential(fock, sigma, -.25, grid, 2., 1e-11)
    np.testing.assert_allclose(chemical, .1, atol=2e-9)
    np.testing.assert_allclose(2*jnp.trace(density), 2., atol=1e-11)
    assert np.isfinite(green).all()
    assert calls == [1], "Eager and implicit scGW should share one canonical charge solver"
