"""Call-local AO reuse separates retained quadrature from primitive scratch.

Tiny analytic GGA fixtures exercise the existing RKS consumer without native
integrals or an XC library. Numerical tests run only in the allocated test step.
"""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.basis import CartesianAO, CartesianBasis
from gradscf.integrals.grids import ao as ao_module, blocks
from gradscf.scf import rks
from gradscf.scf.inputs import grid


@pytest.mark.parametrize('density_dtype,itemsize', [('float64', 8), ('complex128', 16)])
def test_operator_budget_charges_full_ri_exchange_not_occupied_hint(density_dtype, itemsize):
    factors = SimpleNamespace(shape=(3112, 880, 880), size=3112*880*880, dtype=np.dtype('float64'))
    reserve = getattr(grid, 'scf_operator_workspace_bytes', lambda **kwargs: 0)(
        df_factors=factors, with_k=True, density_dtype=np.dtype(density_dtype))
    assert reserve >= factors.size*itemsize, 'One full QNN projected product is live even with occupied hints'


def test_operator_budget_pure_ri_j_needs_only_auxiliary_vectors():
    factors = SimpleNamespace(shape=(3112, 880, 880), size=3112*880*880, dtype=np.dtype('float64'))
    reserve = getattr(grid, 'scf_operator_workspace_bytes', lambda **kwargs: factors.size*8)(
        df_factors=factors, with_k=False, density_dtype=np.dtype('float64'))
    assert 0 < reserve <= 4*3112*8, 'RI-J must not reserve an unused QNN exchange product'


def test_operator_budget_reuses_native_estimate_for_ri_j_direct_k(monkeypatch):
    from gradscf.integrals.backends.native import jk
    top = SimpleNamespace(angular_momenta=(0,), primitive_counts=(1,), contraction_counts=(1,),
                          nuclear_charges=(1,), cart=True)
    provider = jk.NativeDirectBasis(SimpleNamespace(topology=top), None)
    calls = []
    def estimate(bound, workers, cutoff):
        calls.append((bound, workers, cutoff))
        return 123456
    monkeypatch.setattr(jk, 'native_jk_workspace_bytes', estimate)
    factors = SimpleNamespace(shape=(3112, 880, 880), size=3112*880*880, dtype=np.dtype('float64'))
    reserve = getattr(grid, 'scf_operator_workspace_bytes', lambda **kwargs: 0)(
        df_factors=factors, direct_basis=provider, only_dfj=True, with_k=True,
        density_dtype=np.dtype('complex128'), workers=64, cutoff=1e-12)
    assert calls == [(provider, 64, 1e-12)]
    assert 3*123456 <= reserve <= 3*123456 + 4*3112*16


@pytest.mark.parametrize('provider', [object(), SimpleNamespace(plan=SimpleNamespace(topology=SimpleNamespace(nao=880)))])
def test_operator_budget_unknown_direct_provider_is_not_zero(provider):
    reserve = getattr(grid, 'scf_operator_workspace_bytes', lambda **kwargs: 0)(
        direct_basis=provider, with_k=True)
    assert reserve is None, 'Unknown opaque work must disable the optional cache'


def quadrature(parameters):
    basis = CartesianBasis(aos=(
        CartesianAO(center=jnp.array([-.3, .1, .2]) + parameters[0] * jnp.array([.3, -.2, .1]),
            angular=(0, 0, 0), exponents=jnp.linspace(.5, 1.2, 8) * jnp.exp(parameters[1]),
            coefficients=jnp.linspace(.08, .12, 8) * (1. + parameters[2])),
        CartesianAO(center=jnp.array([.4, -.2, .1]), angular=(1, 0, 0),
            exponents=jnp.linspace(.7, 1.1, 8), coefficients=jnp.full(8, .1))))
    points = jnp.stack([jnp.linspace(-.9, 1.1, 17), jnp.full(17, .12), jnp.full(17, .25)], axis=1)
    points = points + parameters[3] * jnp.array([-.1, .2, .3])
    weights = jnp.linspace(.03, .05, 17) * jnp.exp(parameters[4])
    return basis, points, weights


@pytest.fixture
def controlled_grid(monkeypatch):
    monkeypatch.setattr(grid, '_GRID_AO_INPUT_CACHE', {})
    monkeypatch.setattr(grid, '_active_device_cache_key', lambda: ('cpu', 0))
    capacity = [12000]
    monkeypatch.setattr(grid, 'workspace_bytes', lambda *args, resident_bytes=0, **kwargs:
                        capacity[0] - int(resident_bytes))

    def local_xc(*, rho, grad, weights, return_tau=False, **kwargs):
        energy = jnp.dot(weights, .13 * rho**2 + .03 * jnp.sum(grad**2, axis=1))
        result = energy, .26 * rho, .06 * grad
        return result + (jnp.zeros_like(rho),) if return_tau else result

    monkeypatch.setattr(rks, '_xc_energy_and_potential_on_grid', local_xc)
    with jax.default_device(jax.devices("cpu")[0]):
        yield capacity


def bundle(parameters, *, evaluate=None, requested=0, reserved=0, cached=False):
    basis, points, weights = quadrature(parameters)
    builder = grid._cached_grid_ao_input_bundle if cached else grid._build_grid_ao_input_bundle
    raw = builder(basis_from_molecule_spec=lambda *args, **kwargs: basis,
        build_molecular_grid_from_spec=lambda *args, **kwargs: (points, weights),
        evaluate_cartesian_ao_with_derivatives=evaluate or ao_module.evaluate_cartesian_ao_with_derivatives,
        spec=SimpleNamespace(symbols=('H',), coords_bohr=np.zeros((1, 3))), basis='toy',
        max_l=1, grids_level=3, needs_ao_laplacian=False, grid_block_size=requested,
        grid_max_memory=None, max_memory=1., scf_max_cycle=0, fixed_workspace_bytes=reserved)
    prepare = getattr(grid, 'prepare_scf_grid', lambda provider, **kwargs: provider)
    prepared = prepare(raw.grid_ao, max_memory=1., resident_bytes=reserved, deriv=1)
    return replace(raw, grid_ao=prepared)


def consume(data, density, model_scale=1.):
    energy, potential = rks._restricted_xc_energy_and_matrix(density=density, ao=data.ao,
        ao_deriv1=data.ao_deriv1, weights=data.grid_weights, cfg=rks.RKSConfig(xc_spec='pbe'),
        xc_kind='GGA', grid_ao=data.grid_ao)
    return model_scale * energy, model_scale * potential


def observe_ao(monkeypatch):
    # Instrumentation changes the Python producer, not its numeric signature.
    # Retrace this one program so another fixture's observer is not reused.
    materialize = getattr(blocks, '_materialize_scf_grid', None)
    if hasattr(materialize, 'clear_cache'):
        materialize.clear_cache()
    calls = []
    original = blocks._evaluate_cartesian_ao_kernel

    def observed(points, *args, **kwargs):
        jax.debug.callback(lambda count: calls.append(int(count)), jnp.asarray(points.shape[0]), ordered=True)
        return original(points, *args, **kwargs)

    def evaluate(basis, points, **kwargs):
        jax.debug.callback(lambda count: calls.append(int(count)), jnp.asarray(points.shape[0]), ordered=True)
        return ao_module.evaluate_cartesian_ao_with_derivatives(basis, points, **kwargs)

    monkeypatch.setattr(blocks, '_evaluate_cartesian_ao_kernel', observed)
    return calls, evaluate


def synchronize(data):
    jax.block_until_ready((data.ao, data.ao_deriv1, data.ao_laplacian, data.grid_ao))


def test_auto_prepares_bounded_ao_once_for_repeated_rks_consumers(monkeypatch, controlled_grid):
    calls, evaluate = observe_ao(monkeypatch)
    parameters, density = jnp.zeros(5), jnp.array([[1., .12], [.12, .7]])
    data = bundle(parameters, evaluate=evaluate)
    synchronize(data)
    assert calls, 'Auto mode must prepare reusable AO blocks when payload plus one block fits'
    assert max(calls) < 17, 'Preparation must not evaluate all primitive channels at once'
    prepared = len(calls)
    first = jax.block_until_ready(consume(data, density))
    second = jax.block_until_ready(consume(data, 1.07 * density))
    assert len(calls) == prepared, 'SCF density updates must reuse AO, not recompute it'
    assert not np.allclose(first[0], second[0]), 'Only AO may be cached, not density-dependent XC'


@pytest.mark.parametrize('capacity,reserved,requested', [(4000, 0, 0), (12000, 8000, 0), (12000, 0, 3)])
def test_low_budget_or_explicit_streaming_keeps_on_demand_ao(monkeypatch, controlled_grid,
                                                           capacity, reserved, requested):
    controlled_grid[0] = capacity
    calls, evaluate = observe_ao(monkeypatch)
    data = bundle(jnp.zeros(5), evaluate=evaluate, requested=requested, reserved=reserved)
    synchronize(data)
    assert not calls, 'Unfunded or explicitly streamed AO must not be retained eagerly'
    assert data.grid_ao is not None
    if requested: assert data.grid_ao.block_size == requested
    jax.block_until_ready(consume(data, jnp.array([[1., .12], [.12, .7]])))
    assert calls and max(calls) < 17


def test_large_call_local_payload_is_not_retained_by_global_grid_cache(monkeypatch, controlled_grid):
    monkeypatch.setattr(grid, '_GRID_AO_INPUT_CACHE_MAXBYTES', 1024)
    calls, evaluate = observe_ao(monkeypatch)
    data = bundle(jnp.zeros(5), evaluate=evaluate, cached=True)
    synchronize(data)
    assert calls, 'The current call still needs its budgeted prepared AO'
    # This toy's AO+deriv payload alone exceeds 1024 bytes. Raw grid/basis
    # metadata may be reused globally, but prepared AO must remain call-local.
    for retained in grid._GRID_AO_INPUT_CACHE.values():
        assert retained.ao.size == retained.ao_deriv1.size == 0
        for leaf in jax.tree_util.tree_leaves(retained.grid_ao):
            if hasattr(leaf, 'shape') and leaf.ndim >= 2:
                assert not (leaf.shape[-1] == 2 and leaf.size >= 17 * 2)


def test_builtin_rks_prepares_one_call_local_cache_before_initial_fock(monkeypatch, controlled_grid):
    calls, _ = observe_ao(monkeypatch)
    basis, points, weights = quadrature(jnp.zeros(5))
    raw = blocks.AOGrid.from_basis(basis, points, weights, block_size=3, adaptive=True)
    original_leaves = tuple(id(value) for value in jax.tree_util.tree_leaves(raw))
    result = rks.run_rks_from_integrals(overlap=jnp.eye(2),
        hcore=jnp.array([[-1., .05], [.05, .6]]), eri=jnp.zeros((2, 2, 2, 2)),
        nelectron=2, nuclear_repulsion=0., ao=jnp.zeros((0, 2)),
        ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=weights, grid_ao=raw,
        config=rks.RKSConfig(xc_spec='pbe', max_cycle=3, conv_tol=0., max_memory=1.))
    jax.block_until_ready(result)
    assert calls and sum(calls) < 2 * 17, 'Initial, loop and final Fock must share one bounded AO pass'
    assert max(calls) < 17
    assert tuple(id(value) for value in jax.tree_util.tree_leaves(raw)) == original_leaves
    assert not grid._GRID_AO_INPUT_CACHE, 'SCF-local numerical AO must not enter the global input cache'
    assert all(not hasattr(value, 'size') or value.size < 17 * 2
               for value in jax.tree_util.tree_leaves(result)), 'Derived AO must not escape in the result'


def test_prepared_ao_preserves_physical_updates_and_joint_jvp_vjp_fd(controlled_grid):
    density = jnp.array([[1., .12], [.12, .7]])
    parameters = jnp.array([.03, -.02, .01, .04, -.03, .02])
    direction = jnp.array([.2, -.3, .1, .4, -.2, .3])
    ddensity = jnp.array([[.2, -.07], [-.07, -.1]])

    def action(p, d, *, prepared):
        if prepared:
            data = bundle(p)
        else:
            basis, points, weights = quadrature(p)
            values, derivatives = ao_module.evaluate_cartesian_ao_with_derivatives(basis, points, deriv=1)
            data = SimpleNamespace(ao=values, ao_deriv1=derivatives, grid_weights=weights, grid_ao=None)
        energy, potential = consume(data, d, model_scale=jnp.exp(p[5]))
        return jnp.concatenate([energy[None], potential.reshape(-1)])

    cached = jax.jit(lambda p, d: action(p, d, prepared=True))
    dense = jax.jit(lambda p, d: action(p, d, prepared=False))
    value, tangent = jax.jvp(cached, (parameters, density), (direction, ddensity))
    reference, reference_tangent = jax.jvp(dense, (parameters, density), (direction, ddensity))
    np.testing.assert_allclose(value, reference, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(tangent, reference_tangent, atol=3e-10, rtol=3e-10)
    weight = jnp.array([.2, -.3, .1, .4, -.2])
    gradients = jax.grad(lambda p, d: jnp.dot(cached(p, d), weight), argnums=(0, 1))(parameters, density)
    expected = jax.grad(lambda p, d: jnp.dot(dense(p, d), weight), argnums=(0, 1))(parameters, density)
    for got, oracle in zip(gradients, expected):
        np.testing.assert_allclose(got, oracle, atol=3e-10, rtol=3e-10)
    hvp = jax.jvp(jax.grad(lambda p, d: jnp.dot(cached(p, d), weight), argnums=(0, 1)),
                  (parameters, density), (direction, ddensity))[1]
    dense_hvp = jax.jvp(jax.grad(lambda p, d: jnp.dot(dense(p, d), weight), argnums=(0, 1)),
                        (parameters, density), (direction, ddensity))[1]
    for got, oracle in zip(hvp, dense_hvp):
        np.testing.assert_allclose(got, oracle, atol=3e-9, rtol=3e-9)
    for step in (1e-4, 3e-5):
        fd = (cached(parameters + step * direction, density + step * ddensity)
              - cached(parameters - step * direction, density - step * ddensity)) / (2 * step)
        np.testing.assert_allclose(tangent, fd, atol=3e-8, rtol=3e-7)
    changed = cached(parameters + .05 * direction, density + .02 * ddensity)
    assert not np.allclose(value, changed), 'Same-shape geometry/basis/model inputs must not be frozen'
    np.testing.assert_allclose(changed, dense(parameters + .05 * direction, density + .02 * ddensity),
                               atol=3e-11, rtol=3e-11)
