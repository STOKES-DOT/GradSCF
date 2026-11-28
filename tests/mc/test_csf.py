"""Frozen G01–G04 CSF graph, fixed-spin action, and density checks."""

from math import comb

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.mc.casci import solve_casci
from gradscf.mc.csf import CSFGraph, make_csf_space
from gradscf.solvers import EigenSolverConfig, LinearOperator
from gradscf.solvers.eigen import solve_hermitian


def model(norb):
    p, q = np.ogrid[:norb, :norb]
    i = np.arange(2 * norb)[:, None, None]
    h = np.diag(-1.2 + 0.43 * np.arange(norb)) + 0.11 / (1 + p + q)
    factors = 0.07 * np.cos(1 + (i + 1) * (p + q + 2))
    factors += 0.03 / (1 + np.abs(p - q) + i)
    return h, np.einsum("Ppq,Prs->pqrs", factors, factors)


SECTORS = [
    (2, 2, 0, 3), (2, 2, 2, 1), (3, 3, 1, 8), (3, 3, 3, 1),
    (4, 4, 0, 20), (4, 4, 2, 15), (4, 4, 4, 1),
]


@pytest.mark.parametrize("norb,nelec,spin2,count", SECTORS)
def test_g01_rank_unrank(norb, nelec, spin2, count):
    graph = CSFGraph.build(norb, nelec, spin2)
    na, nb = (nelec + spin2) // 2, (nelec - spin2) // 2
    assert graph.dimension == count == comb(norb, na) * comb(norb, nb) - (
        comb(norb, na + 1) * comb(norb, nb - 1) if nb else 0
    )
    assert [graph.rank(graph.unrank(i)) for i in range(count)] == list(range(count))
    with pytest.raises(IndexError):
        graph.unrank(count)


@pytest.mark.parametrize("norb,nelec,spin2", [(2, 2, 1), (2, 4, 2), (3, (2, 0), 0)])
def test_g01_invalid_sector(norb, nelec, spin2):
    with pytest.raises(ValueError):
        CSFGraph.build(norb, nelec, spin2)


def _hop(bit, p, q):
    if not bit & (1 << q):
        return bit, 0
    sign = (-1) ** (bit & ((1 << q) - 1)).bit_count()
    bit ^= 1 << q
    if bit & (1 << p):
        return bit, 0
    sign *= (-1) ** (bit & ((1 << p) - 1)).bit_count()
    return bit | (1 << p), sign


def _determinant_operators(graph):
    """Independent fixed-M determinant oracle in interleaved spin-orbital order."""
    n = graph.norb
    na, nb = graph.nelec
    a_mask = sum(1 << (2 * p) for p in range(n))
    bits = [
        bit for bit in range(1 << (2 * n))
        if (bit & a_mask).bit_count() == na and (bit & ~a_mask).bit_count() == nb
    ]
    address = {state: i for i, state in enumerate(bits)}
    e = np.zeros((2, n, n, len(bits), len(bits)))
    for s, p, q, i in np.ndindex(2, n, n, len(bits)):
        state, phase = _hop(bits[i], 2 * p + s, 2 * q + s)
        if phase:
            e[s, p, q, address[state], i] = phase

    prefix_s2 = []
    for end in range(1, n + 1):
        mask = (1 << (2 * end)) - 1
        z = np.array([
            ((state & mask & a_mask).bit_count() -
             (state & mask & ~a_mask).bit_count()) / 2 for state in bits
        ])
        s2 = np.diag(z * (z + 1))
        for i, p, q in np.ndindex(len(bits), end, end):
            raised, a = _hop(bits[i], 2 * p, 2 * p + 1)
            lowered, b = _hop(raised, 2 * q + 1, 2 * q)
            if a and b:
                s2[address[lowered], i] += a * b
        prefix_s2.append(s2)

    # Project a canonical spin string onto every prefix Casimir eigenspace.
    # Its chosen signs fix the phase in the maximal-M representation.
    vectors = []
    decompositions = [np.linalg.eigh(s2) for s2 in prefix_s2]
    for index in range(graph.dimension):
        path = graph.unrank(index)
        state = sum(int(step) << (2 * p) for p, step in enumerate(path))
        vector = np.eye(len(bits))[:, address[state]]
        spin2 = 0
        for step, (values, frame) in zip(path, decompositions):
            spin2 += (step == 1) - (step == 2)
            valid = np.abs(values - spin2 * (spin2 + 2) / 4) < 1e-10
            sub = frame[:, valid]
            vector = sub @ (sub.T @ vector)
        vectors.append(vector / np.linalg.norm(vector))
    return np.stack(vectors, axis=1), e, prefix_s2[-1]


def _oracle(graph, h, g):
    q, e_spin, s2 = _determinant_operators(graph)
    e = e_spin.sum(axis=0)
    normal_two = np.einsum("pqik,rskj->pqrsij", e, e)
    normal_two -= np.einsum("qr,psij->pqrsij", np.eye(graph.norb), e)
    h_det = np.einsum("pq,pqij->ij", h, e) + 0.5 * np.einsum(
        "pqrs,pqrsij->ij", g, normal_two
    )
    return q, h_det, e_spin, normal_two, s2


@pytest.mark.parametrize("norb,nelec,spin2,count", SECTORS)
def test_g02_g03_spin_sector_action_and_densities(norb, nelec, spin2, count):
    space = make_csf_space(norb, nelec, spin2)
    h, g = model(norb)
    q, h_det, e_spin, two, s2 = _oracle(space.graph, h, g)
    projection = q.T @ h_det @ q
    h_csf = space.matmat(h, g, jnp.eye(count))
    np.testing.assert_allclose(q.T @ q, np.eye(count), atol=1e-12, rtol=0)
    np.testing.assert_allclose(h_csf, projection, atol=2e-11, rtol=0)
    np.testing.assert_allclose(h_csf, h_csf.T, atol=2e-12, rtol=0)
    np.testing.assert_allclose(space.diagonal(h, g), np.diag(projection), atol=2e-11, rtol=0)
    values, vectors = np.linalg.eigh(h_csf)
    np.testing.assert_allclose(values, np.linalg.eigvalsh(projection), atol=1e-10, rtol=0)
    c = vectors[:, 0]
    c_det = q @ c
    d, gamma = space.rdm12(c)
    da, db = space.rdm1s(c)
    spins = np.einsum("i,spqij,j->spq", c_det, e_spin, c_det)
    np.testing.assert_allclose([da, db], spins, atol=1e-10, rtol=0)
    np.testing.assert_allclose(d, spins.sum(axis=0), atol=1e-10, rtol=0)
    np.testing.assert_allclose(gamma, np.einsum("i,pqrsij,j->pqrs", c_det, two, c_det),
                               atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.trace(d), nelec, atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.einsum("pqrr->pq", gamma), (nelec - 1) * d,
                               atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        np.sum(h * d) + 0.5 * np.sum(g * gamma), values[0], atol=1e-10, rtol=0,
    )
    np.testing.assert_allclose(c_det @ s2 @ c_det, spin2 * (spin2 + 2) / 4,
                               atol=1e-10, rtol=0)


def test_g04_large_drt_without_enumerating_coefficients():
    graph = CSFGraph.build(12, 12, 0)
    assert graph.dimension == 226512
    indices = [k * (graph.dimension - 1) // 31 for k in range(32)]
    assert [graph.rank(graph.unrank(i)) for i in indices] == indices
    print(f"CAS(12,12) DRT: {graph.node_count=} {graph.arc_count=}")
    assert graph.node_count < graph.dimension
    assert graph.arc_count < graph.dimension


def test_g04_matrix_free_action_on_cas88():
    space = make_csf_space(8, 8, 0)
    h, g = (jnp.asarray(x) for x in model(8))
    c = jnp.sin(1 + jnp.arange(space.size, dtype=h.dtype))
    c /= jnp.linalg.norm(c)
    action = jax.jit(lambda h, g, c: space.matvec(h, g, c))
    lowered = action.lower(h, g, c).as_text()
    assert f"{space.size}x{space.size}x" not in lowered
    assert "70x70" not in lowered
    assert "4900x" not in lowered
    sigma = action(h, g, c)
    d, gamma = space.rdm12(c)
    np.testing.assert_allclose(jnp.dot(c, sigma), jnp.sum(h * d) + .5 * jnp.sum(g * gamma),
                               atol=1e-10, rtol=0)
    # The graph action is self-adjoint for symmetric input integrals.
    probe = jnp.roll(c, 1)
    np.testing.assert_allclose(jnp.dot(probe, sigma), jnp.dot(c, action(h, g, probe)),
                               atol=1e-10, rtol=0)


def test_g04_davidson_residual():
    space = make_csf_space(4, 4, 0)
    h, g = (jnp.asarray(x) for x in model(4))
    op = LinearOperator(
        shape=(space.size, space.size), dtype=h.dtype,
        matvec=lambda v: space.matvec(h, g, v),
        diagonal=space.diagonal(h, g),
    )
    roots = solve_hermitian(op, config=EigenSolverConfig(method="davidson", atol=1e-10))
    assert bool(roots.converged[0])
    assert float(roots.residual_norms[0]) <= 1e-9


def test_g04_action_budget_checked_before_enumerating_paths(monkeypatch):
    graph = CSFGraph.build(12, 12, 0)

    def forbidden_unrank(*args, **kwargs):
        raise AssertionError("action budget must be checked before path enumeration")

    monkeypatch.setattr(CSFGraph, "unrank", forbidden_unrank)
    with pytest.raises(ValueError, match="CSF.*memory budget"):
        make_csf_space(graph.norb, graph.nelec, graph.spin2, max_memory=1.0)


@pytest.mark.parametrize("pair_block_size", [None, 3])
def test_symmetric_pair_sigma_matches_ordered_projection_and_response(pair_block_size):
    space = make_csf_space(4, (2, 1), 1)
    h, g = map(jnp.asarray, model(4))
    c = jnp.sin(1. + jnp.arange(space.size))
    probe = jnp.roll(c, 1)

    def project_and_apply(h1, eri, vector):
        h1 = .5 * (h1 + h1.T)
        eri = .5 * (eri + eri.transpose(1, 0, 2, 3))
        eri = .5 * (eri + eri.transpose(0, 1, 3, 2))
        eri = .5 * (eri + eri.transpose(2, 3, 0, 1))
        return space.matvec(h1, eri, vector)

    packed = jax.jit(space.matvec_symmetric) if pair_block_size is None else jax.jit(
        lambda h, g, c: space.matvec_symmetric(h, g, c, pair_block_size=pair_block_size)
    )
    expected, expected_pullback = jax.vjp(project_and_apply, h, g, c)
    actual, actual_pullback = jax.vjp(packed, h, g, c)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    for result, target in zip(actual_pullback(probe), expected_pullback(probe)):
        np.testing.assert_allclose(result, target, atol=2e-12, rtol=0)
    tangents = (jnp.sin(h), jnp.cos(g), jnp.cos(c))
    _, d_expected = jax.jvp(project_and_apply, (h, g, c), tangents)
    _, d_actual = jax.jvp(packed, (h, g, c), tangents)
    np.testing.assert_allclose(d_actual, d_expected, atol=2e-12, rtol=0)

    shape = packed.lower(h, g, c).as_text()
    width = 10 if pair_block_size is None else pair_block_size
    assert f"tensor<{width}x{space.size}xf64>" in shape
    if pair_block_size is not None:
        assert f"tensor<10x{space.size}xf64>" not in shape
    assert f"tensor<16x{space.size}xf64>" not in shape
    assert space.action_workspace_bytes("float64", symmetric=True) < space.action_workspace_bytes("float64")



def test_casci_uses_pair_tiles_when_full_action_exceeds_budget():
    space = make_csf_space(4, (2, 1), 1)
    h, g = map(jnp.asarray, model(4))
    config = EigenSolverConfig(method="dense")
    reference = solve_casci(h, g, space, config=config, with_response=False)
    limited = jax.jit(lambda h, g: solve_casci(
        h, g, space, config=config, max_memory=0.063, with_response=False
    ))(h, g)
    assert bool(limited.converged)
    np.testing.assert_allclose(limited.active_energy, reference.active_energy, atol=2e-12, rtol=0)
    np.testing.assert_allclose(limited.ci_residual, reference.ci_residual, atol=2e-12, rtol=0)


def test_pair_tiling_bounds_compiled_cas88_action_scratch():
    space = make_csf_space(8, 8, 0)
    h, g = map(jnp.asarray, model(8))
    c = jnp.sin(1. + jnp.arange(space.size))
    full = jax.jit(space.matvec_symmetric).lower(h, g, c).compile()
    tiled = jax.jit(
        lambda h, g, c: space.matvec_symmetric(h, g, c, pair_block_size=14)
    ).lower(h, g, c).compile()
    np.testing.assert_allclose(tiled(h, g, c), full(h, g, c), atol=2e-11, rtol=0)
    assert tiled.memory_analysis().temp_size_in_bytes < full.memory_analysis().temp_size_in_bytes
