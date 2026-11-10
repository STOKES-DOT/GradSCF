"""Closed shared orbitals reuse the existing one-density RKS Fock."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace

from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.scf import orbital_optimization as orbital
from gradscf.scf.uks import UKSConfig


@pytest.mark.parametrize("xc", ["hf", "svwn", "pbe", "pbe0", "wb97xd"])
def test_closed_direct_objective_and_response_use_one_density(monkeypatch, xc):
    topology, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 1.4", "sto-3g", unit="Bohr", cart=True)
    provider = NativeDirectBasis(integrals.make_plan(topology), parameters)
    occupations = ((1., 0.), (1., 0.))
    config = UKSConfig(xc_spec=xc, jk_backend="direct")
    _, rotate, density_from, evaluate, _ = orbital._problem_functions(
        "roks", xc, "col", occupations, uks_config=config)
    _, _, _, unrestricted, _ = orbital._problem_functions(
        "uks", xc, "col", occupations, uks_config=config)
    ao = jnp.array([[.7, .1], [.2, .8], [0., 0.]])
    args = dict(hcore=jnp.array([[-1., .13], [.13, .2]]), nuclear_repulsion=jnp.asarray(.3),
        eri=None, direct_basis=provider, grid_ao=None, ao=ao,
        ao_deriv1=jnp.stack([ao, .1*ao, -.13*ao, .05*ao]), grid_weights=jnp.array([.3, .4, .1]))
    if xc == "wb97xd":
        args.update(eri_lr=None, dispersion_energy=jnp.asarray(-.003))

    def objective(angle, callback):
        density = density_from(rotate(angle[None], jnp.eye(2)[None]))
        return callback(density, args)[0]

    point = jnp.asarray(.19)
    expected, gradient = jax.jit(jax.value_and_grad(lambda a: objective(a, unrestricted)))(point)
    expected_hvp = jax.jit(jax.grad(jax.grad(lambda a: objective(a, unrestricted))))(point)
    original, calls = NativeDirectBasis.get_jk, []

    def record(self, density, **kwargs):
        calls.append((density.shape, kwargs.get("with_k", True)))
        return original(self, density, **kwargs)

    monkeypatch.setattr(NativeDirectBasis, "get_jk", record)
    actual, actual_gradient = jax.jit(jax.value_and_grad(lambda a: objective(a, evaluate)))(point)
    actual_hvp = jax.jit(jax.grad(jax.grad(lambda a: objective(a, evaluate))))(point)
    np.testing.assert_allclose([actual, actual_gradient, actual_hvp], [expected, gradient, expected_hvp],
                               atol=3e-11, rtol=3e-11)
    assert calls and all(shape == (2, 2) for shape, _ in calls), calls
    if xc in {"svwn", "pbe"}:
        assert all(not with_k for _, with_k in calls)
    # Energy-only AD must keep its scalar kernel, without tracing an unused
    # potential (especially costly for orbital optimization and nested AD).
    from gradscf.scf import rks
    def forbidden(*args, **kwargs):
        raise AssertionError("Energy-only evaluation must not construct an unused RKS potential")
    monkeypatch.setattr(rks, "_energy_and_raw_fock_for_density", forbidden)
    density = density_from(rotate(point[None], jnp.eye(2)[None]))
    np.testing.assert_allclose(evaluate(density, args, compute_potential=False)[0], expected,
                               atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("method,occupations", [("roks", ((1., 0.), (0., 0.))),
                                               ("uks", ((1., 0.), (1., 0.)))])
def test_open_or_independent_spins_retain_batched_direct_jk(monkeypatch, method, occupations):
    topology, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 1.4", "sto-3g", unit="Bohr", cart=True)
    provider = NativeDirectBasis(integrals.make_plan(topology), parameters)
    _, _, density_from, evaluate, _ = orbital._problem_functions(
        method, "hf", "col", occupations, uks_config=UKSConfig(xc_spec="hf", jk_backend="direct"))
    original, shapes = NativeDirectBasis.get_jk, []

    def record(self, density, **kwargs):
        shapes.append(density.shape)
        return original(self, density, **kwargs)

    monkeypatch.setattr(NativeDirectBasis, "get_jk", record)
    args = dict(hcore=jnp.diag(jnp.array([-1., .5])), eri=None, direct_basis=provider,
        nuclear_repulsion=0., ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)),
        grid_weights=jnp.zeros(0), grid_ao=None)
    coeff = jnp.broadcast_to(jnp.eye(2), (1 if method == "roks" else 2, 2, 2))
    assert np.isfinite(float(evaluate(density_from(coeff), args)[0]))
    assert shapes == [(2, 2, 2)]


def test_opaque_direct_provider_retains_its_original_contract():
    shapes = []
    def get_jk(density):
        shapes.append(density.shape)
        return .3*density, .1*density
    provider = SimpleNamespace(nao=2, get_jk=get_jk)
    occupations = ((1., 0.), (1., 0.))
    _, _, density_from, evaluate, _ = orbital._problem_functions(
        "roks", "hf", "col", occupations, uks_config=UKSConfig(xc_spec="hf", jk_backend="direct"))
    value, fock = evaluate(density_from(jnp.eye(2)[None]), dict(
        hcore=jnp.diag(jnp.array([-1., .5])), eri=None, direct_basis=provider,
        nuclear_repulsion=0., ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)),
        grid_weights=jnp.zeros(0), grid_ao=None))
    assert np.isfinite(float(value)) and np.isfinite(np.asarray(fock)).all()
    assert shapes == [(2, 2, 2)]
