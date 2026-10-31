"""Only certified classic Fock callbacks may bypass density-energy AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import orbital_optimization as orbital
from gradscf.scf.uks import UKSConfig


def _classic_problem(xc, empty_beta=False, potential_clip=None):
    occupations = ((1., 0., 0.), (0., 0., 0.) if empty_beta else (1., 0., 0.))
    functions = orbital._problem_functions("uks", xc, "col", occupations,
        uks_config=UKSConfig(xc_spec=xc, potential_clip=potential_clip))
    ao = jnp.array([[.8, .1, -.2], [.2, .6, .1], [-.1, .2, .7],
                    [0., 0., 0.], [1e-8, -2e-8, 1e-8]])
    derivative = jnp.stack([ao, jnp.full_like(ao, .11), .07*ao, jnp.full_like(ao, -.09)])
    factors = jnp.array([[[.5, .1, 0.], [.1, .4, -.03], [0., -.03, .3]],
                         [[.1, -.02, .04], [-.02, .2, .03], [.04, .03, .25]]])
    eri = jnp.einsum("Qpq,Qrs->pqrs", factors, factors)
    args = dict(ao=ao, ao_deriv1=derivative, grid_weights=jnp.array([.3, .2, .4, .1, .2]),
        hcore=jnp.array([[-1., .1, .03], [.1, .4, .04], [.03, .04, .9]]),
        eri=eri, nuclear_repulsion=jnp.asarray(.2), grid_ao=None, direct_basis=None)
    if xc == "wb97xd":
        args.update(eri_lr=.35*eri, dispersion_energy=jnp.asarray(-.003))
    alpha = jnp.array([[.8, .1, .02], [.1, .6, -.03], [.02, -.03, .4]])
    density = jnp.stack([alpha, jnp.zeros_like(alpha) if empty_beta else .3*alpha])
    return occupations, functions, args, density


@pytest.mark.parametrize("xc", ["hf", "svwn", "pbe", "pbe0", "wb97xd"])
@pytest.mark.parametrize("empty_beta", [False, True], ids=["both-spins", "empty-beta"])
def test_classic_reported_fock_matches_density_gradient(xc, empty_beta):
    _, (_, _, _, evaluate, _), args, density = _classic_problem(xc, empty_beta)
    rho = jnp.einsum("gp,spq,gq->g", args["ao"], density, args["ao"])
    assert np.all(np.asarray(rho[:3]) > 1e-12) and np.all(np.asarray(rho[3:]) < 1e-12)
    value, reported = evaluate(density, args)
    energy = lambda d: evaluate(.5*(d+jnp.swapaxes(d, -1, -2)), args, compute_potential=False)[0]
    expected_value, expected_fock = jax.jit(jax.value_and_grad(energy))(density)
    assert np.isfinite(np.asarray(reported)).all() and np.isfinite(np.asarray(expected_fock)).all()
    np.testing.assert_allclose(value, expected_value, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(reported, expected_fock, atol=3e-9, rtol=3e-9)


def test_clipped_classic_callback_keeps_density_ad_fallback(monkeypatch):
    occupations, (dimension, rotate, density_from, evaluate, energy), args, _ = _classic_problem(
        "pbe", potential_clip=0.)
    coeff = jnp.stack([jnp.eye(3)]*2)
    density = density_from(coeff)
    expected = jax.grad(energy)(jnp.zeros(dimension), coeff, args)
    reported = evaluate(density, args)[1]
    physical = jax.grad(lambda d: evaluate(.5*(d+jnp.swapaxes(d, -1, -2)),
                                          args, compute_potential=False)[0])(density)
    assert float(jnp.linalg.norm(reported-physical)) > 1e-4
    original_grad, calls = jax.grad, []

    def record_grad(function, *a, **kw):
        calls.append(True)
        return original_grad(function, *a, **kw)

    monkeypatch.setattr(jax, "grad", record_grad)
    actual = orbital._brillouin_gradient("uks", occupations, coeff, density_from, evaluate, args)
    assert calls
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-9)


@pytest.mark.parametrize("method", ["roks", "uks"])
def test_known_variational_callback_avoids_density_grad(monkeypatch, method):
    occupations = ((1., 0.), (1., 0.))
    dimension, rotate, density_from = orbital._rotation_functions(method, occupations)
    coeff = rotate(jnp.full(dimension, .17),
                   jnp.broadcast_to(jnp.eye(2), (1 if method == "roks" else 2, 2, 2)))

    def evaluate(density, args, *, compute_potential=True):
        h = jnp.array([[-1., .15+args["parameter"]], [.15+args["parameter"], .4]])
        value = jnp.sum(h*density)+.1*jnp.sum(density**2)+.07*jnp.sum(density[0]*density[1])
        fock = h+.2*density+.07*density[::-1]
        return value, fock if compute_potential else None

    def reference(parameter):
        return jax.grad(lambda angles: evaluate(density_from(rotate(angles, coeff)),
            {"parameter": parameter}, compute_potential=False)[0])(jnp.zeros(dimension))

    point, direction = jnp.asarray(.1), jnp.asarray(1.)
    expected, expected_jvp = jax.jvp(reference, (point,), (direction,))

    def forbidden(*args, **kwargs):
        raise AssertionError("A certified variational Fock must not create another density gradient")

    monkeypatch.setattr(jax, "grad", forbidden)
    action = lambda parameter: orbital._brillouin_gradient(
        method, occupations, coeff, density_from, evaluate, {"parameter": parameter},
        known_variational=True)
    actual, actual_jvp = jax.jvp(action, (point,), (direction,))
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(actual_jvp, expected_jvp, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("source", ["full", "native", "projected", "unknown"])
def test_only_canonical_direct_minimum_certifies_fock(monkeypatch, source):
    from gradscf import integrals, scf
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis

    @jax.tree_util.register_pytree_node_class
    class NoncanonicalDirect(NativeDirectBasis):
        pass

    topology, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 1.4", "sto-3g",
                                                    unit="Bohr", cart=True)
    native = NativeDirectBasis(integrals.make_plan(topology), parameters)
    providers = dict(native=native, projected=ProjectedNativeDirectBasis(native, jnp.eye(2)),
                     unknown=NoncanonicalDirect(native.plan, parameters))
    original, calls = orbital._brillouin_gradient, []

    def record(*args, **kwargs):
        calls.append(kwargs.get("known_variational", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(orbital, "_brillouin_gradient", record)
    state = scf.minimize_roks_from_integrals(
        overlap=jnp.eye(2), hcore=jnp.diag(jnp.array([-1., .5])),
        eri=native.plan.evaluate("eri", parameters) if source == "full" else None,
        direct_basis=providers.get(source), nuclear_repulsion=0., ao=jnp.zeros((0, 2)),
        ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
        mo_coeff=jnp.eye(2), mo_occ=np.array([[1., 0.], [1., 0.]]), max_iterations=0)
    assert np.isfinite(float(state.total_energy))
    assert calls and all(value is (source in {"native", "projected"}) for value in calls), calls
