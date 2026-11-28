"""Fixed A01–A03 active-space integral fixtures, in Hartree and Bohr."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from gradscf import fci, integrals


CORE = (0, 3)
ACTIVE = (4, 1, 5)
BLOCKS = ("paaa", "pqaa")


def _pack(eri):
    rows, cols = np.tril_indices(eri.shape[0])
    return eri[rows[:, None], cols[:, None], rows[None, :], cols[None, :]]


@pytest.fixture(scope="module")
def synthetic():
    p, q = np.indices((7, 7))
    auxiliary = np.arange(14)[:, None, None]
    h = np.diag(-1.2 + .43 * np.arange(7)) + .11 / (1 + p + q)
    factors = .07 * np.cos(1 + (auxiliary + 1) * (p + q + 2))
    factors += .03 / (1 + np.abs(p - q) + auxiliary)
    u = 1.0 / (1 + np.arange(7))
    overlap = np.eye(7) + .03 * np.outer(u, u)
    lower = np.where(p > q, .09 * np.sin(1 + p + q), 0.)
    rotation = expm(lower - lower.T)
    eri = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    return tuple(map(jnp.asarray, (overlap, h, eri, rotation, .27, factors)))


def _frame(overlap, rotation):
    return jax.scipy.linalg.solve_triangular(
        jnp.linalg.cholesky(overlap).T, rotation, lower=False
    )


def _oracle(hcore, eri, coeff, enuc, core, active):
    g = jnp.einsum(
        "pqrs,pi,qj,rk,sl->ijkl", eri, coeff, coeff, coeff, coeff,
        precision="highest", optimize="optimal",
    )
    folded = fci.fold_core(coeff.T @ hcore @ coeff, g, core=core, active=active, ecore=enuc)
    a = jnp.asarray(active)
    p = jnp.arange(coeff.shape[1])
    return *folded, (g[jnp.ix_(p, a, a, a)], g[jnp.ix_(p, p, a, a)])


def _check_tree(actual, expected, atol):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=atol, rtol=0.)


def test_a01_ordered_active_block_core_folding(synthetic):
    from gradscf.mc.integrals import transform_active, transform_active_blocks

    s, h, g, u, enuc, factors = synthetic
    c = _frame(s, u)
    expected = _oracle(h, g, c, enuc, CORE, ACTIVE)
    for source in ({"eri": g}, {"eri_pair_matrix": _pack(g)}, {"df_factors": factors}):
        actual = transform_active_blocks(
            h, c, core=CORE, active=ACTIVE, ecore=enuc, blocks=BLOCKS, **source,
        )
        _check_tree(actual, expected, 2e-11)
        plain = transform_active(h, c, core=CORE, active=ACTIVE, ecore=enuc, **source)
        _check_tree(plain, expected[:3], 2e-11)
    electronic = transform_active(h, c, core=CORE, active=ACTIVE, eri=g)[2]
    np.testing.assert_allclose(expected[2] - electronic, enuc, atol=2e-11, rtol=0.)


@pytest.fixture(scope="module")
def lih():
    pyscf = pytest.importorskip("pyscf")
    geometry = "Li 0 0 0; H 0 0 1.6"
    mol = pyscf.gto.M(atom=geometry, basis="sto-3g", cart=False, verbose=0)
    mf = mol.RHF()
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged
    topology, parameters = integrals.prepare_basis(geometry, "sto-3g", cart=False)
    plan = integrals.make_plan(topology)
    c = jnp.asarray(mf.mo_coeff)
    np.testing.assert_allclose(
        c.T @ plan.evaluate("overlap", parameters) @ c, np.eye(topology.nao),
        atol=2e-10, rtol=0.,
    )
    df = pyscf.df.incore.cholesky_eri(mol, auxbasis="weigend")
    return plan, parameters, c, jnp.asarray(pyscf.lib.unpack_tril(df))


def test_a02_lih_native_batch_uses_selected_output(lih, monkeypatch):
    from gradscf.ao2mo import native
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    from gradscf.mc.integrals import transform_active_blocks

    plan, parameters, c, factors = lih
    h = plan.evaluate("kinetic", parameters) + plan.evaluate("nuclear", parameters)
    g = plan.evaluate("eri", parameters)
    enuc = 3. / jnp.linalg.norm(parameters.nuclear_coords[0] - parameters.nuclear_coords[1])
    core, active = (0,), (1, 2)
    expected = _oracle(h, g, c, enuc, core, active)
    seen = []
    slabs = []
    evaluate_slab = native.evaluate_eri_slab
    transform = native.transform_eri_blocks
    evaluate = type(plan).evaluate

    def selected(spaces, requests, direct_basis, **kwargs):
        seen.append(tuple(tuple(spaces[i].shape[1] for i in request) for request in requests))
        return transform(spaces, requests, direct_basis, **kwargs)

    def slab(atm, bas, env, coords, bounds, shape, **kwargs):
        slabs.append(shape)
        return evaluate_slab(atm, bas, env, coords, bounds, shape, **kwargs)

    def checked(self, operator, *args, **kwargs):
        assert operator != "eri", "Selected native production must avoid full AO ERIs"
        return evaluate(self, operator, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Selected native production must avoid a full AO-pair matrix")

    monkeypatch.setattr(native, "transform_eri_blocks", selected)
    monkeypatch.setattr(native, "evaluate_eri_slab", slab)
    monkeypatch.setattr(type(plan), "evaluate", checked)
    monkeypatch.setattr(NativeDirectBasis, "response_eri_pair_matrix", forbidden)
    for source in ({"eri": g}, {"eri_pair_matrix": _pack(g)},
                   {"direct_basis": NativeDirectBasis(plan, parameters)}):
        result = transform_active_blocks(
            h, c, core=core, active=active, ecore=enuc, blocks=BLOCKS, **source,
        )
        _check_tree(result, expected, 2e-10)
    assert seen == [((2, 2, 2, 2), (c.shape[1], 2, 2, 2), (c.shape[1], c.shape[1], 2, 2))]
    assert slabs and all(np.prod(shape) < c.shape[0] ** 4 for shape in slabs)
    reconstructed = jnp.einsum("Lpq,Lrs->pqrs", factors, factors)
    fitted = transform_active_blocks(
        h, c, core=core, active=active, ecore=enuc, blocks=BLOCKS, df_factors=factors,
    )
    _check_tree(fitted, _oracle(h, reconstructed, c, enuc, core, active), 2e-10)


def _normalized_direction(value, symmetry=None):
    direction = jnp.sin(jnp.arange(value.size) + 1.).reshape(value.shape)
    if symmetry == "matrix":
        direction = .5 * (direction + direction.T)
    elif symmetry == "eri":
        direction = .25 * (direction + direction.swapaxes(0, 1)
                            + direction.swapaxes(2, 3) + direction.transpose(1, 0, 3, 2))
        direction = .5 * (direction + direction.transpose(2, 3, 0, 1))
    elif symmetry == "df":
        direction = .5 * (direction + direction.swapaxes(1, 2))
    return direction / jnp.linalg.norm(direction)


def _projection(outputs):
    leaves = jax.tree.leaves(outputs)
    length = sum(value.size for value in leaves)
    weights = jnp.cos(jnp.arange(length) + 1.)
    weights = weights / jnp.linalg.norm(weights)
    offset = 0
    result = jnp.asarray(0.)
    for value in leaves:
        result = result + value.ravel() @ weights[offset:offset + value.size]
        offset += value.size
    return result


def _assert_derivatives(loss, oracle, point, tangent):
    tangent_value = jax.jvp(loss, point, tangent)[1]
    gradient = jax.grad(loss, argnums=tuple(range(len(point))))(*point)
    adjoint_value = sum(jnp.vdot(v, d) for v, d in zip(gradient, tangent, strict=True))
    np.testing.assert_allclose(tangent_value, adjoint_value, atol=2e-9, rtol=0.)
    for step in (2e-4, 1e-4):
        plus = [v + step * d for v, d in zip(point, tangent, strict=True)]
        minus = [v - step * d for v, d in zip(point, tangent, strict=True)]
        fd = (oracle(*plus) - oracle(*minus)) / (2 * step)
        np.testing.assert_allclose(tangent_value, fd, atol=2e-6, rtol=0.)


@pytest.mark.parametrize("representation", ("eri", "eri_pair_matrix", "df_factors"))
def test_a03_array_jvp_matches_adjoint_and_reconstruction(synthetic, representation):
    from gradscf.mc.integrals import transform_active_blocks

    s, h, g, rotation, enuc, factors = synthetic
    rep, sym = {
        "eri": (g, "eri"), "eri_pair_matrix": (_pack(g), "matrix"),
        "df_factors": (factors, "df"),
    }[representation]
    point = (s, h, rep, rotation, enuc)
    tangents = tuple(_normalized_direction(v, sym) for v, sym in zip(
        point, ("matrix", "matrix", sym, None, None), strict=True))
    rows, cols = np.tril_indices(s.shape[0])
    indices = np.empty(s.shape, dtype=int)
    indices[rows, cols] = indices[cols, rows] = np.arange(rows.size)

    def loss(s, h, data, u, e):
        return _projection(transform_active_blocks(
            h, _frame(s, u), core=CORE, active=ACTIVE, ecore=e, blocks=BLOCKS,
            **{representation: data},
        ))

    def oracle(s, h, data, u, e):
        if representation == "eri_pair_matrix":
            data = data[indices[:, :, None, None], indices[None, None, :, :]]
        elif representation == "df_factors":
            data = jnp.einsum("Lpq,Lrs->pqrs", data, data)
        return _projection(_oracle(h, data, _frame(s, u), e, CORE, ACTIVE))

    _assert_derivatives(loss, oracle, point, tangents)


def test_a03_lih_native_jvp_for_coordinates_and_coefficients(lih):
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    from gradscf.mc.integrals import transform_active_blocks

    plan, parameters, c, _ = lih
    core, active = (0,), (1, 2)
    # Every shell follows its atom. STO-3G Li uses the first three shells.
    shell_to_atom = jnp.asarray([0, 0, 0, 1])
    coords = parameters.nuclear_coords
    np.testing.assert_allclose(parameters.centers, coords[shell_to_atom], atol=0., rtol=0.)

    def evaluate(r, c, *, native):
        moving = replace(parameters, centers=r[shell_to_atom], nuclear_coords=r)
        h = plan.evaluate("kinetic", moving) + plan.evaluate("nuclear", moving)
        e = 3. / jnp.linalg.norm(r[0] - r[1])
        if native:
            result = transform_active_blocks(
                h, c, core=core, active=active, ecore=e, blocks=BLOCKS,
                direct_basis=NativeDirectBasis(plan, moving),
            )
        else:
            result = _oracle(h, plan.evaluate("eri", moving), c, e, core, active)
        return _projection(result)

    point = (coords, c)
    loss = lambda r, coeff: evaluate(r, coeff, native=True)
    oracle = lambda r, coeff: evaluate(r, coeff, native=False)
    for selected in range(len(point)):
        tangent = tuple(_normalized_direction(value) if i == selected else jnp.zeros_like(value)
                        for i, value in enumerate(point))
        _assert_derivatives(loss, oracle, point, tangent)


def test_transition_core_action_retains_nonsymmetric_density(synthetic):
    from gradscf.ao2mo.active import core_potential

    s, _, g, u, _, factors = synthetic
    frame = _frame(s, u)
    bra = frame[:, jnp.asarray(CORE)]
    ket = bra + .03 * frame[:, jnp.asarray((2, 6))]
    density = 2 * ket @ bra.T
    j = jnp.einsum("pqrs,rs->pq", g, density)
    k = jnp.einsum("prqs,rs->pq", g, density)
    expected = j - .5 * k
    assert np.max(np.abs(np.asarray(expected - expected.T))) > 1e-5
    for source in ({"eri": g}, {"eri_pair_matrix": _pack(g)}, {"df_factors": factors}):
        result = core_potential(density, source, symmetric=False)
        np.testing.assert_allclose(result, expected, atol=2e-11, rtol=0.)


def test_s4_reuses_identical_pair_products_and_preserves_cotangents(monkeypatch):
    from gradscf.ao2mo import blocks

    c = jnp.sin(jnp.arange(6, dtype=jnp.float64)).reshape(3, 2)
    d = jnp.cos(jnp.arange(3, dtype=jnp.float64))[:, None]
    factors = jnp.arange(18, dtype=c.dtype).reshape(3, 6) * .03
    packed = factors.T @ factors
    original = blocks._mo_pair_products
    calls = []

    def record(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(blocks, "_mo_pair_products", record)

    def block(a, b, e, f):
        return blocks.transform_eri_block((a, b, e, f), eri_pair_matrix=packed)

    def shared(a, b):
        return block(a, b, a, b)

    value = shared(c, d)
    assert len(calls) == 1
    calls.clear()
    # The copies here preserve values while supplying independent JAX inputs.
    c2, d2 = jnp.array(c, copy=True), jnp.array(d, copy=True)
    expected, reference = jax.vjp(block, c, d, c2, d2)
    assert len(calls) == 2
    np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0.)
    probe = jnp.cos(expected)
    a, b, e, f = reference(probe)
    _, pullback = jax.vjp(shared, c, d)
    actual = pullback(probe)
    np.testing.assert_allclose(actual[0], a + e, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(actual[1], b + f, atol=1e-12, rtol=0.)


@pytest.mark.parametrize("large_first", [True, False])
def test_asymmetric_packed_transform_under_budget(large_first):
    from gradscf.ao2mo import blocks

    nao = 12
    p, q = np.tril_indices(nao)
    ids = np.empty((nao, nao), dtype=np.int32)
    ids[p, q] = ids[q, p] = np.arange(p.size, dtype=np.int32)
    # Deliberately nonsymmetric across the two AO pairs. Reverse routing must
    # preserve the actual packed argument's ordering as well as its cotangent.
    raw = jnp.sin(jnp.arange(p.size**2, dtype=jnp.float64).reshape(p.size, p.size)) * .03
    rng = np.random.default_rng(113)
    shape = (11, 10, 1, 1) if large_first else (1, 1, 11, 10)
    coeffs = tuple(jnp.asarray(rng.normal(size=(nao, n)) * .1) for n in shape)

    def candidate(c, d, e, f, matrix):
        return blocks.transform_eri_block((c, d, e, f), eri_pair_matrix=matrix, max_memory=.12)

    def reference(c, d, e, f, matrix):
        full = matrix[ids[:, :, None, None], ids[None, None, :, :]]
        return jnp.einsum("ijkl,ip,jq,kr,ls->pqrs", full, c, d, e, f, precision="highest")

    operands = (*coeffs, raw)
    expected, expected_pullback = jax.vjp(reference, *operands)
    actual, actual_pullback = jax.vjp(jax.jit(candidate), *operands)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0.)
    cotangent = jnp.cos(expected)
    for a, b in zip(actual_pullback(cotangent), expected_pullback(cotangent)):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=0.)
    directions = tuple(jnp.cos(x) for x in operands)
    actual_jvp = jax.jvp(candidate, operands, directions)[1]
    expected_jvp = jax.jvp(reference, operands, directions)[1]
    np.testing.assert_allclose(actual_jvp, expected_jvp, atol=2e-12, rtol=0.)


def test_asymmetric_packed_transform_preserves_mixed_precision():
    from gradscf.ao2mo import blocks

    nao = 8
    p, q = map(jnp.asarray, np.tril_indices(nao))
    rng = np.random.default_rng(113)
    wide = tuple(jnp.asarray(rng.normal(size=(nao, 7))) for _ in range(2))
    small = tuple(jnp.asarray(rng.normal(size=(nao, 1)), dtype=jnp.float32) for _ in range(2))
    packed = jnp.asarray(rng.normal(size=(p.size, p.size)), dtype=jnp.float32)

    def reference(c, d, e, f, matrix):
        left = blocks._mo_pair_products(c, d, p, q)
        right = blocks._mo_pair_products(e, f, p, q)
        return jnp.einsum("pqP,PQ,rsQ->pqrs", left, matrix, right, precision="highest")

    def transform(c, d, e, f, matrix):
        return blocks.transform_eri_block((c, d, e, f), eri_pair_matrix=matrix)

    point = (*wide, *small, packed)
    tangent = tuple(jnp.cos(x) for x in point)
    expected = jax.jvp(jax.jit(reference), point, tangent)
    actual = jax.jvp(jax.jit(transform), point, tangent)
    assert actual[0].dtype == jnp.float64
    _check_tree(actual, expected, 2e-12)
