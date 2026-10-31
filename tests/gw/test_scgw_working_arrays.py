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


def test_canonical_mu_resolves_energy_when_particle_residual_is_already_small():
    """A small charge residual alone does not resolve mu in a cold gap."""
    from gradscf.gw.matsubara import matsubara_grid
    from gradscf.gw.scgw_response import _chemical_potential

    beta = 20.
    grid = matsubara_grid(nw=8, beta=beta)
    fock = jnp.diag(jnp.array([-.5, .7]))
    sigma = jnp.zeros((16, 2, 2), dtype=jnp.complex128)
    occupation = 1 / (1 + np.exp(beta * .6))
    susceptibility = 4 * beta * occupation * (1 - occupation)
    # This guess already passes the 1e-12-electron gate, while its energy
    # error is about 8e-10 Ha. At N=2 the exact mu is the level midpoint.
    guess = .1 + .4e-12 / susceptibility

    chemical, valid = _chemical_potential(fock, sigma, guess, grid, 2., 1e-12)
    assert bool(valid)
    np.testing.assert_allclose(chemical, .1, atol=2e-11, rtol=0)

    direction = jnp.diag(jnp.array([.3, -.2]))
    for step in (3e-4, 1e-4):
        plus, plus_valid = _chemical_potential(fock + step * direction, sigma, guess, grid, 2., 1e-12)
        minus, minus_valid = _chemical_potential(fock - step * direction, sigma, guess, grid, 2., 1e-12)
        assert bool(plus_valid) and bool(minus_valid)
        finite = (plus - minus) / (2 * step)
        np.testing.assert_allclose(finite, .05, atol=3e-7, rtol=0)
