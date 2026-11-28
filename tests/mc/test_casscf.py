"""Frozen CASSCF molecular and projected Super-CI references."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.mc.csf import make_csf_space
from gradscf.mc.orbitals import (
    _lowest_super_ci_vector, _rotation_pairs, _super_ci_model, solve_casscf,
)
from gradscf.mc.casci import solve_casci
from gradscf.mc.integrals import transform_active


CASES = {
    "H2-R": ("H 0 0 0; H 0 0 1.4", "6-31g", 0, (), (0, 1), (1, 1),
             -1.051606680216211, -1.065638305056641),
    "LiH-C": ("Li 0 0 0; H 0 0 1.6", "sto-3g", 0, (0,), (1, 2), (1, 1),
              -7.862128833372132, -7.881045251314651),
    "H3-D": ("H 0 0 0; H 0 0 .90; H .85 0 .31", "6-31g", 1, (), (0, 1, 2), (2, 1),
             -1.510677772022706, -1.524105411664464),
}


def molecular_case(name):
    pyscf = pytest.importorskip("pyscf")
    atoms, basis, spin, core, active, electron_counts, casci, casscf = CASES[name]
    mol = pyscf.gto.M(atom=atoms, basis=basis, spin=spin, symmetry=False, verbose=0)
    mf = (mol.ROHF() if spin else mol.RHF()).run(
        conv_tol=1e-13, conv_tol_grad=1e-9, max_cycle=100,
    )
    assert mf.converged
    overlap, hcore, coefficients, enuc, eri = map(jnp.asarray, (
        mf.get_ovlp(), mf.get_hcore(), mf.mo_coeff, mol.energy_nuc(),
        mol.intor("int2e", aosym="s1"),
    ))
    operators = make_csf_space(len(active), electron_counts, spin)
    return mol, (overlap, hcore, coefficients, enuc), dict(
        operators=operators, core=core, active=active, eri=eri,
    ), casci, casscf


@pytest.mark.parametrize("name", CASES)
def test_casscf_molecular_references(name):
    _, inputs, options, casci, casscf = molecular_case(name)
    coefficients = np.array(inputs[2])
    out = solve_casscf(*inputs, **options)
    assert out.converged
    np.testing.assert_allclose(out.total_energy, casscf, atol=1e-8, rtol=0)
    assert out.ci_residual <= 1e-9
    assert np.linalg.norm(out.orbital_gradient, ord=np.inf) <= 1e-7
    assert abs(out.energy_change) <= 1e-10
    np.testing.assert_allclose(out.mo_coeff.T @ inputs[0] @ out.mo_coeff,
                               jnp.eye(coefficients.shape[1]), atol=1e-10, rtol=0)
    np.testing.assert_array_equal(inputs[2], coefficients)
    h, g, const = transform_active(inputs[1], inputs[2], core=options["core"],
                                    active=options["active"], ecore=inputs[3],
                                    eri=options["eri"])
    initial = solve_casci(h, g, options["operators"], ecore=const)
    np.testing.assert_allclose(initial.total_energy, casci, atol=2e-9, rtol=0)
    if name == "H2-R":
        assert float(initial.total_energy - out.total_energy) > 0.01


def _excite(vector, strings, p, q):
    """Independent Slater determinant oracle for E[p,q]."""
    out = np.zeros_like(vector)
    lookup = {int(bits): i for i, bits in enumerate(strings)}
    for i, bitstring in enumerate(strings):
        bits = int(bitstring)
        if bits & (1 << q) and not bits & (1 << p):
            lo, hi = sorted((p, q))
            n = (bits & ((1 << hi) - (1 << (lo + 1)))).bit_count()
            out[lookup[bits ^ (1 << q) ^ (1 << p)], :] += (-1)**n * vector[i, :]
    return out


def test_exact_super_ci_projected_lih():
    pyscf = pytest.importorskip("pyscf")
    _, (s, h, c, enuc), options, _, _ = molecular_case("LiH-C")
    core, active, ops = options["core"], options["active"], options["operators"]
    h_a, g_a, e_c = transform_active(h, c, core=core, active=active, ecore=enuc,
                                       eri=options["eri"])
    state = solve_casci(h_a, g_a, ops, ecore=e_c)
    d1, d2 = ops.rdm12(state.ci_coeff)
    pairs = _rotation_pairs(c.shape[1], core, active)
    metric, coupling, action = _super_ci_model(s, h, c, d1, d2, core, active, pairs,
                                               eri=options["eri"])
    sigma = np.asarray(jax.vmap(action, in_axes=1, out_axes=1)(jnp.eye(len(pairs))))
    n = c.shape[1]
    nspin = 2
    strings = pyscf.fci.cistring.make_strings(range(n), nspin)
    rows = {int(bits): i for i, bits in enumerate(strings)}
    e_a, ci_a = pyscf.fci.direct_spin1.kernel(np.asarray(h_a), np.asarray(g_a), 2, (1, 1))
    psi = np.zeros((len(strings),) * 2)
    for i, p in enumerate(active):
        for j, q in enumerate(active):
            psi[rows[(1 << core[0]) | (1 << p)], rows[(1 << core[0]) | (1 << q)]] = ci_a[i, j]
    vectors = [_excite(psi, strings, p, q) + _excite(psi.T, strings, p, q).T
               for p, q in pairs]
    hmo = np.asarray(c.T @ h @ c)
    gmo = np.einsum("pqrs,pa,qb,rc,sd->abcd", options["eri"], c, c, c, c, optimize=True)
    absorbed = pyscf.fci.direct_spin1.absorb_h1e(hmo, gmo, n, (nspin, nspin), .5)
    def apply(v):
        return pyscf.fci.direct_spin1.contract_2e(absorbed, v, n, (nspin, nspin))
    e_ref = float(e_a) + float(e_c) - float(enuc)
    projected = [np.asarray(apply(v)) - e_ref * v for v in vectors]
    np.testing.assert_allclose(metric, [[np.vdot(w, v) for v in vectors] for w in vectors],
                               atol=2e-10, rtol=0)
    hp = apply(psi)
    np.testing.assert_allclose(coupling, [np.vdot(w, hp) for w in vectors], atol=2e-9, rtol=0)
    np.testing.assert_allclose(sigma, [[np.vdot(w, v) for v in projected] for w in vectors],
                               atol=2e-9, rtol=0)


def test_casscf_diagnostics():
    _, inputs, opts, _, _ = molecular_case("LiH-C")
    for active in ((0, 1), (1, 1), (1, 20)):
        with pytest.raises(ValueError, match="active|orbital"):
            solve_casscf(*inputs, **(opts | {"active": active}))
    with pytest.raises(ValueError, match="max_cycle"):
        solve_casscf(*inputs, **opts, max_cycle=0)
    partial = solve_casscf(*inputs, **opts, max_cycle=1)
    assert not partial.converged
    assert not partial.response_valid
    ops = make_csf_space(2, (1, 1), 0)
    result = solve_casscf(jnp.eye(2), jnp.zeros((2, 2)), jnp.eye(2), 0., ops,
                          core=(), active=(0, 1), eri=jnp.zeros((2,)*4))
    assert result.converged and not result.response_valid
    assert result.gap <= 1e-8

    # A pure nuclear-energy derivative bypasses the coupled residual, so it
    # must obey the same invalid-state policy at the final observable.
    def degenerate_energy(enuc):
        return solve_casscf(jnp.eye(2), jnp.zeros((2, 2)), jnp.eye(2), enuc, ops,
                            core=(), active=(0, 1), eri=jnp.zeros((2,)*4)).total_energy

    assert np.isnan(jax.grad(degenerate_energy)(0.))


def test_super_ci_guarded_matrix_free_solver():
    """Supply an independent dense oracle for the production Ritz branch."""
    n = 65
    diagonal = jnp.full(n, 0.8)
    couplings = 0.04 * jnp.sin(jnp.arange(n, dtype=diagonal.dtype) + 0.2)
    actual = _lowest_super_ci_vector(couplings, lambda x: diagonal * x, diagonal, 64.)
    reference = np.zeros((n + 1, n + 1))
    reference[0, 1:] = reference[1:, 0] = couplings
    reference[1:, 1:] = np.diag(diagonal + 0.001)
    expected = np.linalg.eigh(reference)[1][:, 0]
    np.testing.assert_allclose(abs(np.vdot(actual, expected)), 1., rtol=0, atol=1e-10)


def test_cas_reference_invariant_under_internal_rotations():
    from scipy.linalg import expm
    nmo = 7
    p = np.arange(nmo)
    i, j = np.meshgrid(p, p, indexing="ij")
    h = np.diag(-1.2 + .43 * p) + .11 / (1 + i + j)
    aux = np.arange(2 * nmo)[:, None, None]
    l = .07 * np.cos(1 + (aux + 1) * (i + j + 2)) + .03 / (1 + np.abs(i-j) + aux)
    g = np.einsum("Ppq,Prs->pqrs", l, l)
    u = 1 / (1 + p)
    s = np.eye(nmo) + .03 * np.outer(u, u)
    k = np.tril(.09 * np.sin(1 + i + j), -1)
    c = np.linalg.solve(np.linalg.cholesky(s).T, expm(k - k.T))
    core, active, outside = (0, 3), (4, 1, 5), (2, 6)
    ops = make_csf_space(3, (1, 1), 0)

    def state(coeff):
        h_a, g_a, e_c = transform_active(h, coeff, core=core, active=active, ecore=.27, eri=g)
        result = solve_casci(h_a, g_a, ops, ecore=e_c)
        c_a = coeff[:, active]
        d1, _ = ops.rdm12(result.ci_coeff)
        return result.total_energy, c_a @ d1 @ c_a.T

    e_ref, d_ref = state(c)
    for group in (core, active, outside):
        rows = np.arange(len(group))
        block = np.tril(.17 * np.sin(1 + rows[:, None] + rows[None, :]), -1)
        spin = expm(block - block.T)
        rotated = c.copy()
        rotated[:, group] = c[:, group] @ spin
        energy, density = state(rotated)
        np.testing.assert_allclose(energy, e_ref, rtol=0, atol=2e-9)
        np.testing.assert_allclose(density, d_ref, rtol=0, atol=2e-8)


def test_super_ci_macro_reuses_density_at_the_same_ci_state():
    _, inputs, options, initial_energy, _ = molecular_case("H2-R")

    class CountDensity:
        def __init__(self, operators):
            self.operators = operators
            self.states = []

        def __getattr__(self, name):
            return getattr(self.operators, name)

        def rdm12(self, coefficients):
            self.states.append(np.asarray(coefficients))
            return self.operators.rdm12(coefficients)

    counter = CountDensity(options["operators"])
    options["operators"] = counter
    with jax.disable_jit():
        result = solve_casscf(*inputs, **options, max_cycle=1, with_response=False)
    assert int(result.iterations) == 1
    assert float(result.total_energy) < initial_energy
    assert len(counter.states) == 2
    np.testing.assert_array_equal(counter.states[-1], result.ci_coeff)


def test_orbital_gradient_pullback_matches_zero_angle_expm_and_response():
    from gradscf.mc.orbitals import _orbital_gradient, _rotation_matrix
    from jax.scipy.linalg import expm

    _, (s, h, c, enuc), options, _, _ = molecular_case("LiH-C")
    core, active, ops = options["core"], options["active"], options["operators"]
    g = options["eri"]
    pairs = _rotation_pairs(c.shape[1], core, active)
    hcas, gcas, offset = transform_active(h, c, core=core, active=active, ecore=enuc, eri=g)
    ci = solve_casci(hcas, gcas, ops, ecore=offset)
    density = ops.rdm12(ci.ci_coeff)
    c = c @ expm(_rotation_matrix(jnp.full(len(pairs), 0.01), pairs, c.shape[1], skew=True))
    def gradient(frame, hcore, eri, d1, d2, *, reference):
        def hamiltonian(x):
            return transform_active(hcore, x, core=core, active=active, ecore=enuc, eri=eri)
        if not reference:
            return _orbital_gradient(frame, (d1, d2), pairs, hamiltonian)
        def energy(angles):
            rotated = frame @ expm(_rotation_matrix(angles, pairs, frame.shape[1], skew=True))
            ha, ga, ec = hamiltonian(rotated)
            return ec + jnp.sum(ha * d1) + .5 * jnp.sum(ga * d2)
        return jax.grad(energy)(jnp.zeros(len(pairs)))
    values = (c, h, g, *density)
    direction = tuple(jnp.sin(x) for x in values)
    actual = jax.jvp(lambda *args: gradient(*args, reference=False), values, direction)
    expected = jax.jvp(lambda *args: gradient(*args, reference=True), values, direction)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0.)


def test_packed_no_core_super_ci_matches_exact_projected_step(monkeypatch):
    from gradscf.mc import orbitals

    # Use non-contiguous active columns to exercise the external-orbital map.
    _, (s, h, c, enuc), options, _, _ = molecular_case("H2-R")
    active = (0, 3)
    eri = options["eri"]
    p, q = np.tril_indices(c.shape[0])
    packed = eri[p[:, None], q[:, None], p, q]
    kwargs = {"eri_pair_matrix": packed, "max_memory": 8000.}
    ops = options["operators"]
    pairs = _rotation_pairs(c.shape[1], (), active)
    hcas, gcas, offset = transform_active(h, c, core=(), active=active, ecore=enuc, **kwargs)
    state = solve_casci(hcas, gcas, ops, ecore=offset)
    d1, d2 = ops.rdm12(state.ci_coeff)
    # The reference uses the exact mixed-AO derivative even on small models.
    ref_b, ref_action = orbitals._super_ci_action(s, h, c, d1, d2, (), active, pairs, kwargs)
    fast_b, fast_action = orbitals._no_core_super_ci_action(h, c, d1, d2, active, kwargs)
    probe = jnp.eye(len(pairs))
    ref_h = jax.vmap(ref_action)(probe)
    fast_h = jax.vmap(fast_action)(probe)
    np.testing.assert_allclose(fast_b, ref_b, atol=2e-10, rtol=0)
    np.testing.assert_allclose(fast_h, ref_h, atol=2e-10, rtol=0)
    ref_step = orbitals._super_ci_step(
        s, h, c, d1, d2, (), active, pairs, {"eri": eri, "max_memory": 8000.}
    )
    def forbid_reference(*args, **kwargs):
        raise AssertionError("packed dense no-core step should reuse the MO blocks")
    monkeypatch.setattr(orbitals, "_super_ci_action", forbid_reference)
    fast_step = orbitals._super_ci_step(s, h, c, d1, d2, (), active, pairs, kwargs)
    np.testing.assert_allclose(fast_step, ref_step, atol=2e-9, rtol=0.)
