"""Batched shell-direct J/K preserves density response under worker partitioning."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import direct_jk


def _context(atom, cart):
    specification = [
        [0, [2.0, 0.8, -0.2], [0.4, 0.3, 1.0]],
        [1, [0.8, 1.0]],
        [2, [0.6, 1.0]],
    ]
    topology, parameters = integrals.prepare_basis(
        atom, specification, cart=cart
    )
    plan = integrals.make_plan(topology)
    atoms, shells, environment = plan.pack(parameters)
    full = plan.evaluate("eri", parameters)
    return topology, atoms, shells, environment, full


@pytest.fixture(params=[False, True], ids=["spherical", "cartesian"])
def basis(request):
    # Uneven shell-pair partitions, general contractions and d-shell AO order.
    return _context("H 0 0 0; H 0.1 0.2 1.1", request.param)


def _action(basis, density, workers, *, cutoff=0.0, **flags):
    topology, atoms, shells, environment, _ = basis
    return direct_jk(
        atoms, shells, environment, density,
        cart=topology.cart, cutoff=cutoff, workers=workers, **flags,
    )


def _full_action(basis, density):
    eri = basis[-1]
    return (
        jnp.einsum("pqrs,...rs->...pq", eri, density),
        jnp.einsum("prqs,...rs->...pq", eri, density),
    )


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("nvec", [4, 20])
def test_partitioned_direct_jk_matches_full_batched_nonsymmetric_density(basis, workers, nvec):
    nao = basis[0].nao
    density = jnp.asarray(np.random.default_rng(801).normal(size=(nvec, nao, nao)))
    observed = jax.jit(lambda d: _action(basis, d, workers))(density)
    for actual, expected in zip(observed, _full_action(basis, density)):
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-10)


def test_partitioned_direct_jk_has_reproducible_worker_reduction(basis):
    nao = basis[0].nao
    density = jnp.asarray(np.random.default_rng(802).normal(size=(4, nao, nao)))
    run = jax.jit(lambda d: _action(basis, d, 4))
    expected = run(density)
    for _ in range(2):
        for actual, reference in zip(run(density), expected):
            np.testing.assert_array_equal(actual, reference)


def test_simultaneous_calls_keep_geometry_and_workspace_ownership_separate(basis):
    second = _context("H 0 0 0; H -0.2 0.3 1.8", basis[0].cart)
    nao = basis[0].nao
    density = jnp.asarray(np.random.default_rng(805).normal(size=(4, nao, nao)))
    actions = [jax.jit(lambda d, context=context: _action(context, d, 4, cutoff=1e-12))
               for context in (basis, second)]
    # Initialize registration and executables before concurrent native calls.
    for action in actions:
        jax.block_until_ready(action(density))
    with ThreadPoolExecutor(max_workers=2) as executor:
        for scale in (1.0, 0.37):
            futures = [executor.submit(lambda action=action: jax.block_until_ready(action(scale * density)))
                       for action in actions]
            for future, context in zip(futures, (basis, second)):
                for actual, expected in zip(future.result(), _full_action(context, scale * density)):
                    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-10)


def test_parallel_schwarz_screening_retains_one_linear_density_map(basis):
    nao = basis[0].nao
    rng = np.random.default_rng(803)
    density, tangent = jnp.asarray(rng.normal(size=(2, 4, nao, nao)))
    screened = lambda d: _action(basis, d, 4, cutoff=1e-6)
    observed = jax.jit(screened)(density + 0.37 * tangent)
    for actual, primal, direction in zip(observed, screened(density), screened(tangent)):
        np.testing.assert_allclose(actual, primal + 0.37 * direction, atol=2e-10, rtol=1e-10)
    for actual, reference in zip(screened(density), _action(basis, density, 1, cutoff=1e-6)):
        np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=1e-10)
    for actual in jax.jit(lambda d: _action(basis, d, 4, cutoff=1e9))(density):
        np.testing.assert_array_equal(actual, 0.0)


def test_parallel_direct_jk_density_jvp_vjp_and_hvp(basis):
    nao = basis[0].nao
    density, tangent = jnp.asarray(
        np.random.default_rng(804).normal(size=(2, 4, nao, nao))
    )
    actual = lambda d: _action(basis, d, 4)
    reference = lambda d: _full_action(basis, d)
    observed = jax.jit(lambda d, t: jax.jvp(actual, (d,), (t,))[1])(density, tangent)
    for result, expected in zip(observed, reference(tangent)):
        np.testing.assert_allclose(result, expected, atol=2e-10, rtol=1e-10)

    def loss(action, d):
        coulomb, exchange = action(d)
        return 0.7 * jnp.sum(coulomb**2) + 0.3 * jnp.sum(exchange**2)

    gradient = jax.grad(lambda d: loss(actual, d))
    reference_gradient = jax.grad(lambda d: loss(reference, d))
    np.testing.assert_allclose(
        jax.jit(gradient)(density), reference_gradient(density), atol=2e-9, rtol=2e-10
    )
    np.testing.assert_allclose(
        jax.jit(lambda d, t: jax.jvp(gradient, (d,), (t,))[1])(density, tangent),
        jax.jvp(reference_gradient, (density,), (tangent,))[1],
        atol=2e-9, rtol=2e-10,
    )


@pytest.mark.parametrize("with_j,with_k", [(True, True), (True, False), (False, True), (False, False)])
def test_direct_selected_operators_keep_nonsymmetric_density_and_ad(basis, with_j, with_k):
    nao = basis[0].nao
    density, tangent = jnp.asarray(
        np.random.default_rng(806).normal(size=(2, 4, nao, nao))
    )
    actual = lambda d: _action(basis, d, 4, with_j=with_j, with_k=with_k)

    def reference(d):
        coulomb, exchange = _full_action(basis, d)
        return (coulomb if with_j else jnp.zeros_like(d),
                exchange if with_k else jnp.zeros_like(d))

    for observed, expected in zip(jax.jit(actual)(density), reference(density)):
        np.testing.assert_allclose(observed, expected, atol=2e-10, rtol=1e-10)
    for observed, expected in zip(
        jax.jit(lambda d, t: jax.jvp(actual, (d,), (t,))[1])(density, tangent),
        reference(tangent),
    ):
        np.testing.assert_allclose(observed, expected, atol=2e-10, rtol=1e-10)

    def loss(action, d):
        coulomb, exchange = action(d)
        return 0.7 * jnp.sum(coulomb**2) + 0.3 * jnp.sum(exchange**2)

    gradient = jax.grad(lambda d: loss(actual, d))
    reference_gradient = jax.grad(lambda d: loss(reference, d))
    np.testing.assert_allclose(jax.jit(gradient)(density), reference_gradient(density),
                               atol=2e-9, rtol=2e-10)
    np.testing.assert_allclose(
        jax.jit(lambda d, t: jax.jvp(gradient, (d,), (t,))[1])(density, tangent),
        jax.jvp(reference_gradient, (density,), (tangent,))[1],
        atol=2e-9, rtol=2e-10,
    )


def test_no_requested_operator_does_not_construct_native_ffi(basis, monkeypatch):
    from gradscf.integrals.backends.native import jk

    def forbidden(*args, **kwargs):
        raise AssertionError("No native J/K operator is needed when both flags are false")

    monkeypatch.setattr(jk, "_direct_operator", forbidden)
    density = jnp.ones((4, basis[0].nao, basis[0].nao))
    observed = jax.jit(lambda d: _action(basis, d, 4, with_j=False, with_k=False))(density)
    for result in observed:
        np.testing.assert_array_equal(result, 0.0)


def test_native_direct_worker_outputs_obey_quota_and_keep_row_fallback():
    # Source guards retain the prohibition on unbounded worker matrices; these
    # are admission/ownership checks, not a measurement of peak process RSS.
    path = Path(__file__).parents[2] / "src/gradscf/integrals/_native/csrc/direct_jk.cc"
    source = re.sub(r"//[^\n]*", "", path.read_text())
    from gradscf.integrals.backends.native.jk import _private_output_limit
    assert _private_output_limit(None) == 64*1024**2
    assert 'private_output_bytes<0' in source
    assert re.search(r"kPrivateBatchLimit\s*=\s*8", source)
    assert "nb<=kPrivateBatchLimit" in source
    assert "size_t(row_workers),size_t(private_output_bytes)/sizeof(double)/size" in source
    assert "private_workers>=2" in source
    assert "private_outputs ? private_workers : row_workers" in source
    assert "private_outputs ? size_t(workers)*size : 0" in source
    assert "private_outputs ? 0 : worker*tables->nbas/workers" in source
    assert "private_outputs ? worker : 0,private_outputs ? workers : 1" in source


@pytest.mark.parametrize("contractions", [1, 128], ids=["small", "cap"])
def test_native_workspace_charges_bounded_small_batch_worker_outputs(monkeypatch, contractions):
    from types import SimpleNamespace
    from gradscf.integrals.backends.native import jk

    # Metadata only, including the large contraction count: no AO/integral
    # array is allocated to test the workspace cap.
    topology = SimpleNamespace(angular_momenta=(0,)*4, primitive_counts=(1,)*4,
        contraction_counts=(contractions,)*4, nuclear_charges=(1,), cart=True)
    provider = SimpleNamespace(plan=SimpleNamespace(topology=topology))
    one = jk.native_jk_workspace_bytes(provider, 1, 0.)
    four = jk.native_jk_workspace_bytes(provider, 4, 0.)
    monkeypatch.setattr(jk, "_DIRECT_JK_PRIVATE_MAX_BYTES", 0)
    assert jk.native_jk_workspace_bytes(provider, 1, 0.) == one
    base = jk.native_jk_workspace_bytes(provider, 4, 0.)
    expected = min(64*1024**2, 4*2*8*(4*contractions)**2*8)
    assert four-base == expected


@pytest.mark.skipif(os.environ.get("GRADSCF_EXPECT_DIAGNOSTICS") != "1",
                    reason="requires the separately instrumented native build")
@pytest.mark.parametrize("with_j,with_k", [(True, True), (True, False), (False, True)],
                         ids=["jk", "j-only", "k-only"])
def test_small_batch_workers_evaluate_each_shell_quartet_once(basis, with_j, with_k,
                                                             record_property):
    from gradscf.integrals._native.diagnostics import get_counts, reset_counts

    nao = basis[0].nao
    density = jnp.asarray(np.random.default_rng(807).normal(size=(2, nao, nao)))
    coulomb, exchange = jax.block_until_ready(_full_action(basis, density))
    expected = (coulomb if with_j else jnp.zeros_like(density),
                exchange if with_k else jnp.zeros_like(density))
    actions = {workers: jax.jit(lambda d, workers=workers: _action(
        basis, d, workers, with_j=with_j, with_k=with_k)) for workers in (1, 4)}
    # Finish compilation, full-ERI work and both warmups before resetting the
    # process-global counters. Count exactly one synchronized native action.
    for action in actions.values():
        jax.block_until_ready(action(density))
    observed, counts = {}, {}
    for workers, action in actions.items():
        reset_counts()
        observed[workers] = jax.block_until_ready(action(density))
        counts[workers] = get_counts()["families"]["eri4c"]["evaluations"]
        record_property(f"workers_{workers}_eri4c_evaluations", counts[workers])
    for result in observed.values():
        for actual, reference in zip(result, expected):
            np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=1e-10)
    for parallel, serial in zip(observed[4], observed[1]):
        np.testing.assert_allclose(parallel, serial, atol=2e-10, rtol=1e-10)
    assert counts[1] > 0
    assert counts[4] == counts[1], (
        f"Small batches must partition shell quartets without repeated evaluation: {counts}")
