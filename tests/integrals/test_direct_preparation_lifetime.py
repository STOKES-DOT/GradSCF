"""A direct provider owns live ENV preparation outside repeated density actions.

Eager count gates synchronize real native J/K calls; they do not establish XLA
loop hoisting or end-to-end speedup. Small dense oracles precede counted work.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native import packing
from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis


@pytest.fixture
def system():
    topology, parameters = integrals.prepare_basis(
        "H 0 0 0; H .2 -.1 .8", "3-21g", cart=True)
    plan = integrals.make_plan(topology)
    density = jnp.asarray(np.random.default_rng(941).normal(size=(topology.nao,) * 2))
    return plan, parameters, density


def _oracle(plan, parameters, *, omega=0.):
    eri = plan.evaluate("eri", parameters, omega=omega)
    return lambda d: (jnp.einsum("pqrs,...rs->...pq", eri, d),
                      jnp.einsum("prqs,...rs->...pq", eri, d))


def _counter(monkeypatch):
    counts = dict(normalize=0, pack=0)
    normalize, pack = packing.normalized_shell_coefficients, packing.pack_environment

    def normalized(*args, **kwargs):
        counts["normalize"] += 1
        return normalize(*args, **kwargs)

    def packed(*args, **kwargs):
        counts["pack"] += 1
        return pack(*args, **kwargs)

    monkeypatch.setattr(packing, "normalized_shell_coefficients", normalized)
    monkeypatch.setattr(packing, "pack_environment", packed)
    return counts


def _assert_jk(observed, expected):
    for actual, reference in zip(jax.block_until_ready(observed), expected):
        np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)


def test_density_actions_prepare_normalization_and_environment_once(system, monkeypatch):
    plan, parameters, density = system
    oracle = _oracle(plan, parameters)
    counts = _counter(monkeypatch)
    provider = NativeDirectBasis(plan, parameters)
    for scale in (1., -.6, 1.3):
        _assert_jk(provider.get_jk(scale * density), oracle(scale * density))
    assert counts == dict(normalize=len(plan.bas), pack=1), (
        "One immutable parameter state must prepare ENV once across eager density updates", counts)


@pytest.mark.parametrize("changed", ["geometry", "exponents", "coefficients", "same_shape_object"])
def test_new_parameter_state_rebinds_without_stale_same_shape_values(system, monkeypatch, changed):
    plan, parameters, density = system
    if changed == "geometry":
        updated = replace(parameters, centers=parameters.centers.at[-1, 2].add(.13))
    elif changed == "exponents":
        updated = replace(parameters, exponents=(parameters.exponents[0] * 1.17,
                                                *parameters.exponents[1:]))
    elif changed == "coefficients":
        coefficients = parameters.coefficients[0].at[0, 0].add(.21)
        updated = replace(parameters, coefficients=(coefficients, *parameters.coefficients[1:]))
    else:
        topology2, updated = integrals.prepare_basis(
            "H 0 0 0; H .2 -.1 1.1", "3-21g", cart=True)
        assert integrals.make_plan(topology2) is plan
    oracle, updated_oracle = _oracle(plan, parameters), _oracle(plan, updated)
    assert any(np.linalg.norm(np.asarray(a-b)) > 1e-6
               for a, b in zip(oracle(density), updated_oracle(density)))
    counts = _counter(monkeypatch)
    original = NativeDirectBasis(plan, parameters)
    rebound = replace(original, parameters=updated)
    for scale in (1., -.4):
        _assert_jk(original.get_jk(scale*density), oracle(scale*density))
        _assert_jk(rebound.get_jk(scale*density), updated_oracle(scale*density))
    assert counts == dict(normalize=2*len(plan.bas), pack=2), (
        "Fresh states bind once; shape caching and density-loop repacking are invalid", counts)


def test_operator_changes_do_not_reuse_a_different_coulomb_environment(system, monkeypatch):
    plan, parameters, density = system
    oracles = {omega: _oracle(plan, parameters, omega=omega) for omega in (0., .37)}
    counts = _counter(monkeypatch)
    provider = NativeDirectBasis(plan, parameters)
    for omega in (0., .37, 0., .37):
        _assert_jk(provider.get_jk(density, omega=omega), oracles[omega](density))
    assert counts == dict(normalize=len(plan.bas), pack=1)


def test_projected_model_update_reuses_only_primitive_preparation(system, monkeypatch):
    plan, parameters, density = system
    oracle = _oracle(plan, parameters)
    counts = _counter(monkeypatch)
    primitive = NativeDirectBasis(plan, parameters)
    transform = jnp.eye(plan.topology.nao) + .07*jnp.sin(density)
    changed = transform + .09*jnp.cos(density)
    outputs = []
    for matrix in (transform, changed):
        provider = ProjectedNativeDirectBasis(primitive, matrix)
        for scale in (1., -.3):
            dm = matrix @ (scale*density) @ matrix.T
            expected = tuple(matrix.T @ value @ matrix for value in oracle(dm))
            result = provider.get_jk(scale*density)
            _assert_jk(result, expected)
            if scale == 1.:
                outputs.append(result)
    assert any(np.linalg.norm(np.asarray(a-b)) > 1e-6 for a, b in zip(*outputs))
    assert counts == dict(normalize=len(plan.bas), pack=1)


def test_provider_pytree_reconstruction_preserves_raw_state_without_array_work(system, monkeypatch):
    plan, parameters, density = system
    oracle = _oracle(plan, parameters)
    provider = NativeDirectBasis(plan, parameters)
    counts = _counter(monkeypatch)
    leaves, structure = jax.tree.flatten(provider)
    restored = jax.tree.unflatten(structure, leaves)
    assert counts == dict(normalize=0, pack=0), "Tree reconstruction must do no array work"
    assert len(leaves) == len(jax.tree.leaves(parameters)), "Only canonical raw state is flattened"
    action = jax.jit(lambda basis, d: basis.get_jk(d))
    _assert_jk(action(restored, density), oracle(density))
    _assert_jk(action(restored, -.2*density), oracle(-.2*density))


def test_bound_environment_retains_geometry_density_mixed_hvp(system):
    plan, parameters, density = system
    centers = parameters.centers
    direction = (.07*jnp.cos(centers), .03*jnp.sin(density))

    def loss(r, d, *, dense):
        current = replace(parameters, centers=r)
        if dense:
            result = _oracle(plan, current)(d)
        else:
            provider = NativeDirectBasis(plan, current)
            result = tuple(a+b for a, b in zip(provider.get_jk(d), provider.get_jk(-.3*d)))
            result = tuple(value/.7 for value in result)
        return sum(jnp.sum(value**2) for value in result)

    gradients = [jax.jit(jax.grad(lambda r, d, dense=dense: loss(r, d, dense=dense), (0, 1)))
                 for dense in (False, True)]
    observed, expected = [jax.jvp(g, (centers, density), direction)[1] for g in gradients]
    _assert_jk(observed, expected)
    for step in (1e-4, 3e-5):
        plus = gradients[0](centers+step*direction[0], density+step*direction[1])
        minus = gradients[0](centers-step*direction[0], density-step*direction[1])
        for value, a, b in zip(observed, plus, minus):
            np.testing.assert_allclose(value, (a-b)/(2*step), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("field", ["exponents", "coefficients"])
def test_prepared_environment_retains_basis_gradient(system, field):
    plan, parameters, density = system
    values = getattr(parameters, field)

    def value(x, *, dense):
        current = replace(parameters, **{field: (x, *values[1:])})
        result = (_oracle(plan, current)(density) if dense else
                  NativeDirectBasis(plan, current).get_jk(density))
        return sum(jnp.sum(v*v) for v in result)

    expected = jax.jit(jax.grad(lambda x: value(x, dense=True)))(values[0])
    actual = jax.jit(jax.grad(lambda x: value(x, dense=False)))(values[0])
    assert float(jnp.linalg.norm(expected)) > 1e-6
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    direction = .07*jnp.cos(jnp.arange(values[0].size).reshape(values[0].shape)+.3)
    for step in (2e-4, 1e-4):
        fd = (value(values[0]+step*direction, dense=True)
              - value(values[0]-step*direction, dense=True))/(2*step)
        np.testing.assert_allclose(jnp.vdot(actual, direction), fd, atol=2e-7, rtol=4e-6)


def test_workspace_quota_change_keeps_parameter_preparation(system, monkeypatch):
    plan, parameters, density = system
    oracle = _oracle(plan, parameters)
    counts = _counter(monkeypatch)
    original = NativeDirectBasis(plan, parameters)
    bounded = replace(original, private_output_bytes=0)
    for provider in (original, bounded, original):
        _assert_jk(provider.get_jk(density), oracle(density))
    assert counts == dict(normalize=len(plan.bas), pack=1)


@pytest.mark.parametrize("field", ["exponents", "coefficients"])
def test_provider_argument_cannot_hide_raw_basis_tangent_in_zero_preparation(system, field):
    plan, parameters, density = system
    provider = NativeDirectBasis(plan, parameters)
    leaves, structure = jax.tree.flatten(provider)
    target = getattr(parameters, field)[0]
    assert any(value is target for value in leaves)
    # Only a raw basis leaf moves. Reconstructed preparation must retain its
    # true tangent, including JAX primitive/contraction normalization.
    tangent_leaves = [jnp.ones_like(value) if value is target else jnp.zeros_like(value)
                      for value in leaves]
    tangent = jax.tree.unflatten(structure, tangent_leaves)
    action = jax.jit(lambda p, dp: jax.jvp(
        lambda q: q.get_jk(density), (p,), (dp,))[1])

    def reference(x):
        current = replace(parameters, **{field: (x, *getattr(parameters, field)[1:])})
        return _oracle(plan, current)(density)

    direction = jnp.ones_like(target)
    expected = jax.jvp(reference, (target,), (direction,))[1]
    assert any(float(jnp.linalg.norm(matrix)) > 1e-6 for matrix in expected)
    actual = action(provider, tangent)
    _assert_jk(actual, expected)
    for step in (2e-4, 1e-4):
        fd = jax.tree.map(lambda plus, minus: (plus-minus)/(2*step),
                          reference(target+step*direction), reference(target-step*direction))
        for observed, finite_difference in zip(actual, fd):
            np.testing.assert_allclose(observed, finite_difference, atol=2e-7, rtol=4e-6)


def test_provider_raw_basis_second_order_is_not_hidden_by_zero_first_tangent(system):
    plan, parameters, density = system
    provider = NativeDirectBasis(plan, parameters)
    leaves, structure = jax.tree.flatten(provider)
    target = parameters.exponents[0]

    def loss(t):
        changed = [value+t*t*jnp.ones_like(value) if value is target else value
                   for value in leaves]
        current = jax.tree.unflatten(structure, changed)
        return sum(jnp.sum(value**2) for value in current.get_jk(density))

    def reference(x):
        current = replace(parameters, exponents=(x, *parameters.exponents[1:]))
        return sum(jnp.sum(value**2) for value in _oracle(plan, current)(density))

    # p(t)=p0+t^2*v has zero first tangent at t=0. Its second derivative is
    # 2*grad_p(E).v, requiring only first basis products, not a basis Hessian.
    expected = 2*jnp.sum(jax.grad(reference)(target))
    assert abs(float(expected)) > 1e-6
    actual = jax.jit(jax.jacfwd(jax.jacfwd(loss)))(0.)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    first = jax.jit(jax.grad(loss))
    np.testing.assert_allclose(first(0.), 0., atol=1e-12)
    for step in (2e-4, 1e-4):
        fd = (first(step)-first(-step))/(2*step)
        np.testing.assert_allclose(actual, fd, atol=2e-7, rtol=4e-6)


def test_mutable_numpy_inputs_are_snapshotted_at_provider_binding(system):
    plan, parameters, density = system
    raw = jax.tree.map(lambda value: np.asarray(value).copy(), parameters)
    expected = _oracle(plan, parameters)(density)
    changed = replace(parameters, exponents=(parameters.exponents[0]*1.17,
                                            *parameters.exponents[1:]))
    changed_expected = _oracle(plan, changed)(density)
    provider = NativeDirectBasis(plan, raw)
    raw.exponents[0][...] *= 1.17
    # A bound provider owns its numerical state. A new provider sees the edit;
    # caller mutation cannot leave its parameters and prepared ENV disagreeing.
    _assert_jk(provider.get_jk(density), expected)
    _assert_jk(NativeDirectBasis(plan, raw).get_jk(density), changed_expected)
    assert all(isinstance(value, jax.Array) for value in jax.tree.leaves(provider.parameters))


def test_changed_angular_topology_rebinds_normalization(system, monkeypatch):
    plan, parameters, density = system
    topology = replace(plan.topology, angular_momenta=(1, *plan.topology.angular_momenta[1:]))
    changed_plan = integrals.make_plan(topology)
    changed_density = jnp.eye(topology.nao)
    expected = _oracle(plan, parameters)(density)
    changed_expected = _oracle(changed_plan, parameters)(changed_density)
    counts = _counter(monkeypatch)
    original = NativeDirectBasis(plan, parameters)
    rebound = replace(original, plan=changed_plan)
    _assert_jk(original.get_jk(density), expected)
    _assert_jk(rebound.get_jk(changed_density), changed_expected)
    _assert_jk(original.get_jk(density), expected)
    assert counts == dict(normalize=2*len(plan.bas), pack=2)
