"""Shell-orbit call counts and independent values for direct/packed Coulomb.

CPU float64; coordinates in Bohr, omega in inverse Bohr, cutoff exactly zero.
The diagnostic gate counts actual libcint evaluations after compilation/warmup.
PySCF supplies the dense oracle, so shared native traversal cannot hide omissions.
"""
from dataclasses import replace
from types import SimpleNamespace
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals._native.diagnostics import get_counts, reset_counts


ATOM = "He 0 0 0; He .3 -.2 1.7"
BASIS = {"He": [[0, [1.3, .7, .2], [.45, .1, .6]],
                 [2, [.9, .6], [.25, .4]]]}


@pytest.fixture(scope="module", params=[True, False], ids=["cartesian", "spherical"])
def problem(request):
    gto = pytest.importorskip("pyscf.gto")
    mol = gto.M(atom=ATOM, basis=BASIS, unit="Bohr", cart=request.param, verbose=0)
    top, parameters = integrals.prepare_basis(ATOM, BASIS, unit="Bohr", cart=request.param)
    plan = integrals.make_plan(top)
    assert len(top.angular_momenta) == mol.nbas == 4
    assert top.nao == mol.nao_nr()
    dense = {}
    for omega in (0., .3):
        with mol.with_range_coulomb(omega):
            dense[omega] = jnp.asarray(mol.intor("int2e", aosym="s1"))
    return SimpleNamespace(top=top, parameters=parameters, plan=plan, mol=mol, dense=dense)


def _dense(tensor, density, selection=(True, True)):
    return (jnp.einsum("pqrs,...rs->...pq", tensor, density) if selection[0] else jnp.zeros_like(density),
            jnp.einsum("prqs,...rs->...pq", tensor, density) if selection[1] else jnp.zeros_like(density))


def _compare(actual, expected, tolerance=1e-10):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


def _count(action, argument, record_property):
    jax.block_until_ready(action(argument))
    reset_counts()
    value = jax.block_until_ready(action(argument))
    report = get_counts()
    assert report["enabled"], "Native count gate requires a --diagnostics build"
    count = report["families"]["eri4c"]["evaluations"]
    record_property("native_library_sha256", report["library_sha256"])
    record_property("eri4c_evaluations", count)
    return value, count


@pytest.mark.skipif(os.getenv("GRADSCF_EXPECT_DIAGNOSTICS") != "1",
                    reason="requires a separate --diagnostics native library")
@pytest.mark.parametrize("workers", [1, 4])
def test_direct_evaluates_each_unordered_shell_pair_pair_once(problem, workers, record_property):
    n = problem.top.nao
    density = jnp.asarray(np.random.default_rng(2026110602).normal(size=(n, n)))
    basis = NativeDirectBasis(problem.plan, problem.parameters)
    action = jax.jit(lambda d: basis.get_jk(d, workers=workers, screening_threshold=0.))
    actual, count = _count(action, density, record_property)
    _compare(actual, _dense(problem.dense[0.], density))
    shells = problem.mol.nbas
    pairs = shells*(shells+1)//2
    expected = pairs*(pairs+1)//2
    record_property("canonical_shell_quartets", expected)
    print("shell-quartet evaluations", problem.top.cart, workers, count, expected, flush=True)
    assert count == expected, f"Repeated shell-pair interchange blocks: {count} != {expected}"


@pytest.mark.skipif(os.getenv("GRADSCF_EXPECT_DIAGNOSTICS") != "1",
                    reason="requires a separate --diagnostics native library")
def test_single_ao_shells_do_not_compute_then_discard_quartets(record_property):
    atom = "He 0 0 0; He .3 -.2 1.7; He -.1 .8 2.9; He .4 .2 4.1"
    specification = {"He": [[0, [.8, 1.]]]}
    top, pars = integrals.prepare_basis(atom, specification, unit="Bohr")
    basis = NativeDirectBasis(integrals.make_plan(top), pars)
    gto = pytest.importorskip("pyscf.gto")
    mol = gto.M(atom=atom, basis=specification, unit="Bohr", verbose=0)
    tensor = jnp.asarray(mol.intor("int2e", aosym="s1"))
    density = jnp.asarray(np.random.default_rng(2026110603).normal(size=(4, 4)))
    actual, count = _count(jax.jit(lambda d: basis.get_jk(d, workers=1)), density, record_property)
    _compare(actual, _dense(tensor, density))
    assert count == 55, f"Four one-AO shells need 55 orbits, observed {count} evaluations"


@pytest.mark.skipif(os.getenv("GRADSCF_EXPECT_DIAGNOSTICS") != "1",
                    reason="requires a separate --diagnostics native library")
@pytest.mark.parametrize("layout", ["s4", "s8"])
def test_packed_eri_uses_the_same_minimal_shell_schedule(problem, layout, record_property):
    action = jax.jit(lambda p: problem.plan.evaluate("eri", p, aosym=layout))
    actual, count = _count(action, problem.parameters, record_property)
    expected = _pack(problem.dense[0.], layout)
    _compare(actual, expected)
    assert count == 55, f"Packed {layout} repeats shell-pair interchange blocks: {count}"


def _pack(tensor, layout):
    n = tensor.shape[0]
    a, b = np.tril_indices(n)
    pairs = tensor[a[:, None], b[:, None], a[None, :], b[None, :]]
    return pairs if layout == "s4" else pairs[np.tril_indices(len(a))]


@pytest.mark.parametrize("workers,quota", [(1, None), (4, None), (4, 0)],
                         ids=["serial", "private-workers", "row-workers"])
@pytest.mark.parametrize("omega", [0., .3])
@pytest.mark.parametrize("selection", [(True, True), (True, False), (False, True)])
def test_repeated_multiao_shells_keep_complex_density_response(problem, workers, quota, omega, selection):
    n = problem.top.nao
    rng = np.random.default_rng(2026110604)
    density = jnp.asarray(rng.normal(size=(2, n, n)) + 1j*rng.normal(size=(2, n, n)))
    direction = jnp.asarray(.1*(rng.normal(size=density.shape) + 1j*rng.normal(size=density.shape)))
    basis = NativeDirectBasis(problem.plan, problem.parameters, private_output_bytes=quota)
    action = lambda d: basis.get_jk(d, workers=workers, omega=omega,
                                    with_j=selection[0], with_k=selection[1])
    reference = lambda d: _dense(problem.dense[omega], d, selection)
    value, tangent = jax.jit(lambda d, v: jax.jvp(action, (d,), (v,)))(density, direction)
    _compare((value, tangent), (reference(density), reference(direction)))
    weights = tuple(jnp.asarray(rng.normal(size=density.shape) + 1j*rng.normal(size=density.shape)) for _ in range(2))
    observed = jax.jit(lambda d: jax.vjp(action, d)[1](weights)[0])(density)
    _compare(observed, jax.vjp(reference, density)[1](weights)[0], 1e-9)
    compiled = jax.jit(action)
    first = jax.block_until_ready(compiled(density))
    second = jax.block_until_ready(compiled(density))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("omega", [0., .3])
def test_shared_derivative_paths_match_independent_geometry_differences(problem, omega):
    n = problem.top.nao
    rng = np.random.default_rng(2026110605)
    density = jnp.asarray(rng.normal(size=(n, n)))
    coordinates = problem.parameters.nuclear_coords
    direction = jnp.asarray([[.03, -.05, .07], [-.02, .04, -.01]])
    # Each atom owns two shells; move their centers with their parent atom.
    def parameters(r):
        return replace(problem.parameters, nuclear_coords=r, centers=jnp.repeat(r, 2, axis=0))
    def direct(r):
        return NativeDirectBasis(problem.plan, parameters(r)).get_jk(density, workers=4, omega=omega)
    def dense_at(r):
        mol = problem.mol.copy()
        mol.set_geom_(np.asarray(r), unit="Bohr")
        with mol.with_range_coulomb(omega):
            return jnp.asarray(mol.intor("int2e", aosym="s1"))
    step = 1e-4
    plus, minus = (dense_at(coordinates + sign*step*direction) for sign in (1, -1))
    finite_difference = (plus-minus)/(2*step)
    expected = _dense(finite_difference, density)
    observed = jax.jit(lambda r: jax.jvp(direct, (r,), (direction,))[1])(coordinates)
    _compare(observed, expected, 3e-8)
    weights = tuple(jnp.asarray(rng.normal(size=(n, n))) for _ in range(2))
    gradient = jax.jit(jax.grad(lambda r: sum(jnp.sum(x*w) for x, w in zip(direct(r), weights))))(coordinates)
    _compare(jnp.vdot(gradient, direction), sum(jnp.sum(x*w) for x, w in zip(expected, weights)), 3e-8)
    for layout in ("s4", "s8"):
        packed = lambda r: problem.plan.evaluate("eri", parameters(r), aosym=layout, omega=omega)
        value, tangent = jax.jit(lambda r: jax.jvp(packed, (r,), (direction,)))(coordinates)
        _compare(value, _pack(problem.dense[omega], layout))
        _compare(tangent, _pack(finite_difference, layout), 3e-8)
