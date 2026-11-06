"""Canonical raw provider state and preparation at real consumer boundaries."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from test_direct_preparation_lifetime import _assert_jk, _counter, _oracle


@pytest.fixture
def system():
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H .2 -.1 .8", "3-21g", cart=True)
    plan = integrals.make_plan(topology)
    density = jnp.asarray(np.random.default_rng(947).normal(size=(topology.nao,)*2))
    return plan, parameters, density


@pytest.mark.parametrize("field", ["exponents", "coefficients", "centers"])
def test_raw_pytree_primal_update_is_never_hidden_by_derived_preparation(system, field):
    plan, parameters, density = system
    provider = NativeDirectBasis(plan, parameters)
    target = getattr(parameters, field)
    original = target if field == "centers" else target[0]
    updated = original.at[0, 0].add(.13) if field != "exponents" else original*1.17
    changed_parameters = replace(parameters, **{
        field: updated if field == "centers" else (updated, *target[1:])})
    expected = _oracle(plan, changed_parameters)(density)
    baseline = _oracle(plan, parameters)(density)
    assert any(np.linalg.norm(np.asarray(x-y)) > 1e-6 for x, y in zip(expected, baseline))
    leaves, structure = jax.tree.flatten(provider)
    changed = jax.tree.unflatten(structure, [
        updated if value is original else value for value in leaves])
    action = jax.jit(lambda p, d: p.get_jk(d))
    _assert_jk(action(provider, density), baseline)
    _assert_jk(action(changed, density), expected)
    _assert_jk(changed.get_jk(density), expected)


def _raw_provider(plan, parameters):
    provider = NativeDirectBasis(plan, parameters)
    leaves, structure = jax.tree.flatten(provider)
    return jax.tree.unflatten(structure, leaves)


@pytest.mark.parametrize("consumer", ["rks_builder", "response_closure"])
def test_reconstructed_raw_state_prepares_once_per_consumer(system, monkeypatch, consumer):
    from gradscf.scf.rks import RKSConfig, _make_jk_builder
    plan, parameters, density = system
    oracle = _oracle(plan, parameters)
    provider = _raw_provider(plan, parameters)
    counts = _counter(monkeypatch)
    if consumer == "rks_builder":
        action = _make_jk_builder(None, RKSConfig(xc_spec="hf", jk_backend="direct"),
                                  direct_basis=provider)
    else:
        action = provider.prepare_jk(cache_bytes=0)
    for scale in (1., -.3, .7):
        _assert_jk(action(scale*density), oracle(scale*density))
    assert counts["normalize"] <= len(plan.bas)
    assert counts["pack"] <= 1, "Preparation must precede this consumer's density loop"


@pytest.mark.parametrize("spin", ["unrestricted", "generalized"])
def test_spin_solver_prepares_reconstructed_state_outside_density_loop(system, monkeypatch, spin):
    from gradscf.scf.uks import UKSConfig, run_uks_from_integrals
    from gradscf.scf.gks import GKSConfig, run_gks_from_integrals
    plan, parameters, _ = system
    overlap = plan.evaluate("overlap", parameters)
    hcore = plan.evaluate("kinetic", parameters) + plan.evaluate("nuclear", parameters)
    eri = plan.evaluate("eri", parameters)
    n = plan.topology.nao
    common = dict(overlap=overlap, hcore=hcore, nuclear_repulsion=0.,
                  ao=jnp.zeros((0, n)), ao_deriv1=jnp.zeros((4, 0, n)),
                  grid_weights=jnp.zeros(0))
    if spin == "unrestricted":
        run = lambda provider, backend: run_uks_from_integrals(
            **common, nalpha=1, nbeta=1, eri=eri if backend == "full" else jnp.zeros((0,)),
            direct_basis=provider, config=UKSConfig(
                xc_spec="hf", jk_backend=backend, max_cycle=60, conv_tol=1e-11))
    else:
        run = lambda provider, backend: run_gks_from_integrals(
            **common, nelectron=2, eri=eri if backend == "full" else None,
            direct_basis=provider, config=GKSConfig(
                xc_spec="hf", jk_backend=backend, max_cycle=60, conv_tol=1e-11))
    expected = run(None, "full")
    provider = _raw_provider(plan, parameters)
    counts = _counter(monkeypatch)
    actual = run(provider, "direct")
    assert bool(actual.converged) and bool(expected.converged)
    np.testing.assert_allclose(actual.total_energy, expected.total_energy, atol=1e-7, rtol=0.)
    assert counts["normalize"] <= len(plan.bas)
    assert counts["pack"] <= 1, "Spin solver must bind before initial/iterative/final density calls"
