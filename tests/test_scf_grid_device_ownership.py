"""Host/native work and device AO storage obey distinct capacity constraints.

Capacity is synthetic; real tiny AO arrays execute on the test CPU. These
tests establish planner arithmetic and values, not GPU runtime performance.
"""
from inspect import signature
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import _memory
from gradscf.integrals.basis import CartesianAO, CartesianBasis
from gradscf.integrals.grids import blocks
from gradscf.integrals.backends.native import jk
from gradscf.scf.inputs import grid


@pytest.fixture
def raw_grid():
    basis = CartesianBasis(aos=(CartesianAO(
        center=jnp.array([.1, -.2, .3]), angular=(0, 0, 0),
        exponents=jnp.array([.7]), coefficients=jnp.array([.9])),))
    points = jnp.stack((jnp.linspace(-1., 1., 8), jnp.full(8, .2), jnp.full(8, -.1)), axis=1)
    return blocks.AOGrid.from_basis(
        basis, points, jnp.linspace(.1, .2, 8), block_size=4, adaptive=True)


@pytest.fixture
def capacity(monkeypatch):
    device = SimpleNamespace(platform="gpu")
    monkeypatch.setattr(grid, "_grid_device", lambda value: device)
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda device=None: 10_000)
    return device


def _prepare(raw, *, host=0, expected_passes=None, **kwargs):
    # The old production callsite adds native-host work to device residents.
    # Exercise that real old behavior until the ownership-aware entry exists.
    parameters = signature(grid.prepare_scf_grid).parameters
    if "host_workspace_bytes" in parameters:
        kwargs["host_workspace_bytes"] = host
    else:
        kwargs["resident_bytes"] = kwargs.get("resident_bytes", 0) + host
    if "expected_passes" in parameters:
        kwargs["expected_passes"] = expected_passes
    return grid.prepare_scf_grid(raw, **kwargs)


def _reservation(**kwargs):
    split = getattr(grid, "_scf_operator_reservation", None)
    if split is not None:
        return split(**kwargs)
    combined = grid.scf_operator_workspace_bytes(**kwargs)
    return None if combined is None else (combined, 0)


def _assert_values(observed, raw):
    actual = observed.block(0, raw.ngrids, deriv=1)
    expected = raw.block(0, raw.ngrids, deriv=1)
    for value, reference in zip(jax.tree.leaves(jax.block_until_ready(actual)),
                                jax.tree.leaves(expected)):
        np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-10)


def test_gpu_native_workspace_deducts_method_budget_before_device_cap(raw_grid, capacity):
    # Usable method=20,000; native host=9,000; usable GPU=8,000.
    # Native host fits the method budget and must not consume GPU capacity.
    prepared = _prepare(raw_grid, max_memory=.025, host=9_000, deriv=1)
    assert prepared is not raw_grid, "Host-native scratch incorrectly consumed GPU capacity"
    assert prepared.block_size == 4
    _assert_values(prepared, raw_grid)


def test_gpu_method_budget_still_limits_tile_after_host_reservation(raw_grid, capacity):
    # Usable method=10,400, leaving 1,400 before device-resident AO leaves.
    prepared = _prepare(raw_grid, max_memory=.013, host=9_000, deriv=1)
    assert prepared is not raw_grid
    owned = sum(int(value.nbytes) for value in jax.tree.leaves(raw_grid))
    available = 10_400 - 9_000 - owned
    point = 4 * 8  # One AO, four channels, float64.
    scratch = 8 * (12 + 2*4 + 8)  # One primitive; same conservative AO contract.
    expected_block = min(4, (available-2*8*point)//(scratch+2*point))
    assert expected_block > 0
    assert prepared.block_size == expected_block
    _assert_values(prepared, raw_grid)


def test_gpu_cannot_drop_host_reservation_from_total_method_budget(raw_grid, capacity):
    assert _prepare(raw_grid, max_memory=.012, host=9_500, deriv=1) is raw_grid


def test_cpu_native_and_ao_storage_share_the_same_capacity(raw_grid, capacity):
    capacity.platform = "cpu"
    assert _prepare(raw_grid, max_memory=.025, host=9_000, deriv=1) is raw_grid


def test_gpu_resident_arrays_still_consume_device_capacity(raw_grid, capacity):
    assert _prepare(raw_grid, max_memory=.025, host=9_000,
                    resident_bytes=7_800, deriv=1) is raw_grid


@pytest.mark.parametrize("passes", [0, 1])
def test_known_single_consumption_keeps_streaming(raw_grid, capacity, monkeypatch, passes):
    def forbidden(*args, **kwargs):
        pytest.fail("A known zero/single consumer must not materialize a reusable AO cache")
    monkeypatch.setattr(grid, "_materialize_scf_grid", forbidden)
    assert _prepare(raw_grid, max_memory=.025, expected_passes=passes) is raw_grid


@pytest.mark.parametrize("passes", [None, 2])
def test_unknown_or_repeated_consumption_preserves_cache_admission(raw_grid, capacity, passes):
    prepared = _prepare(raw_grid, max_memory=.025, expected_passes=passes)
    assert prepared is not raw_grid
    _assert_values(prepared, raw_grid)


def _native_provider(nao=2):
    topology = SimpleNamespace(nao=nao, angular_momenta=(0,)*nao, primitive_counts=(1,)*nao,
                               contraction_counts=(1,)*nao, nuclear_charges=(1,)*nao, cart=True)
    return jk.NativeDirectBasis(SimpleNamespace(topology=topology), None)


@pytest.mark.parametrize("with_ri", [False, True])
def test_native_three_call_reserve_is_host_owned(monkeypatch, with_ri):
    provider = _native_provider()
    factors = SimpleNamespace(shape=(7, 2, 2), dtype=np.dtype("float64")) if with_ri else None
    calls = []
    def estimate(bound, workers, cutoff):
        calls.append((bound, workers, cutoff))
        return 3_000
    monkeypatch.setattr(jk, "native_jk_workspace_bytes", estimate)
    observed = _reservation(df_factors=factors, direct_basis=provider, only_dfj=with_ri,
                            with_k=True, density_dtype=np.dtype("complex128"), workers=4, cutoff=1e-12)
    assert observed == (2*7*16 if with_ri else 0, 9_000)
    assert calls == [(provider, 4, 1e-12)]


@pytest.mark.parametrize("dtype,itemsize", [("float64", 8), ("complex128", 16)])
def test_ri_exchange_full_qnn_stays_on_device(dtype, itemsize):
    factors = SimpleNamespace(shape=(7, 2, 2), dtype=np.dtype("float64"))
    assert _reservation(df_factors=factors, with_k=True, density_dtype=np.dtype(dtype)) == (
        (2*7 + 7*2*2)*itemsize, 0)


@pytest.mark.parametrize("projected", [False, True])
def test_unknown_or_projected_native_reservation_stays_conservative(projected):
    provider = (jk.ProjectedNativeDirectBasis(_native_provider(), jnp.eye(2))
                if projected else object())
    assert _reservation(direct_basis=provider, with_k=True) is None


@pytest.mark.parametrize("spin", ["restricted", "unrestricted", "generalized"])
def test_scf_consumers_forward_host_ownership_before_grid_admission(raw_grid, monkeypatch, spin):
    from gradscf.scf import rks, uks, gks
    provider = _native_provider(nao=1)
    monkeypatch.setattr(jk, "native_jk_workspace_bytes", lambda *args: 3_000)
    class ReachedGrid(Exception):
        pass
    def observe(raw, **kwargs):
        assert kwargs.get("host_workspace_bytes", 0) == 9_000
        assert kwargs["resident_bytes"] < 9_000, "Host reserve must not be duplicated as device-resident"
        raise ReachedGrid
    monkeypatch.setattr(grid, "prepare_scf_grid", observe)
    # Match the one-AO quadrature. No native products are evaluated: stop at
    # the actual solver's grid-admission boundary, after its array accounting.
    common = dict(overlap=jnp.eye(1), hcore=jnp.array([[-1.]]), nuclear_repulsion=0.,
                  eri=jnp.zeros((0,)), ao=jnp.zeros((0, 1)), ao_deriv1=jnp.zeros((4, 0, 1)),
                  grid_weights=raw_grid.weights, grid_ao=raw_grid, direct_basis=provider)
    with pytest.raises(ReachedGrid):
        if spin == "restricted":
            rks.run_rks_from_integrals(**common, nelectron=2,
                config=rks.RKSConfig(xc_spec="pbe", jk_backend="direct", max_cycle=2))
        elif spin == "unrestricted":
            uks.run_uks_from_integrals(**common, nalpha=1, nbeta=1,
                config=uks.UKSConfig(xc_spec="pbe", jk_backend="direct", max_cycle=2))
        else:
            gks.run_gks_from_integrals(**common, nelectron=2,
                config=gks.GKSConfig(xc_spec="pbe", jk_backend="direct", max_cycle=2))
