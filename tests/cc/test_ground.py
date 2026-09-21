"""Real RHF CC regression: CPU/float64, Hartree, independent PySCF oracles."""

from itertools import product
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def h4():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from pyscf.cc import rccsd

    mol = pyscf.gto.M(
        atom="H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1", basis="sto-3g", verbose=0
    )
    mf = mol.RHF().run(conv_tol=1e-12)
    h = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 4)
    ref = rccsd.RCCSD(mf).set(conv_tol=1e-12, conv_tol_normt=1e-10, max_cycle=150).run()
    assert ref.converged
    return mf, h, g, ref


def test_cc_public_namespace():
    import gradscf

    assert "cc" in dir(gradscf)


@pytest.mark.parametrize("model", ["ccsd", "cc2"])
def test_random_amplitude_residual_matches_pyscf(h4, model):
    from gradscf.cc.integrals import prepare_integrals
    from gradscf.cc.rccsd import residual, correlation_energy
    from pyscf.cc import rccsd

    mf, h, g, ref = h4
    eris = ref.ao2mo()
    ref.cc2 = model == "cc2"
    rng = np.random.default_rng(5)
    t1 = 0.02 * rng.normal(size=(2, 2))
    t2 = 0.03 * rng.normal(size=(2, 2, 2, 2))
    t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))
    u1, u2 = rccsd.update_amps(ref, t1, t2, eris)
    d = mf.mo_energy[:2, None] - mf.mo_energy[None, 2:]
    expected = ((u1 - t1) * d, (u2 - t2) * (d[:, None, :, None] + d[None, :, None, :]))
    ints = prepare_integrals(h, g, nocc=2, nuclear_repulsion=mf.mol.energy_nuc())
    actual = jax.jit(lambda a, b: residual(a, b, ints, model=model))(t1, t2)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-12)
    np.testing.assert_allclose(
        correlation_energy(t1, t2, ints), rccsd.energy(ref, t1, t2, eris), atol=1e-12
    )
    ref.cc2 = False


def test_ccsd_energy_amplitudes_and_lambda(h4):
    from gradscf import cc

    mf, h, g, ref = h4
    out = cc.run_cc(
        h,
        g,
        nocc=2,
        nuclear_repulsion=mf.mol.energy_nuc(),
        config=cc.CCConfig(conv_tol=1e-11, residual_tol=1e-10),
    )
    assert out.converged
    np.testing.assert_allclose(out.total_energy, ref.e_tot, atol=2e-9)
    np.testing.assert_allclose(out.t1, ref.t1, atol=2e-8)
    np.testing.assert_allclose(out.t2, ref.t2, atol=2e-8)
    lam = cc.solve_lambda(h, g, out, nocc=2)
    l1, l2 = ref.solve_lambda()
    assert lam.converged
    np.testing.assert_allclose(lam.l1, l1, atol=2e-7)
    np.testing.assert_allclose(lam.l2, l2, atol=2e-7)


@pytest.mark.parametrize("method", ["CCS", "CCD", "LCCD", "LCCSD", "CC2", "CCSD"])
def test_models_converge_to_their_own_residual(h4, method):
    from gradscf import cc
    from gradscf.cc.integrals import prepare_integrals
    from gradscf.cc.rccsd import residual

    mf, h, g, ref = h4
    cfg = cc.CCConfig(method=method, conv_tol=1e-10, residual_tol=1e-9)
    out = cc.run_cc(h, g, nocc=2, config=cfg)
    assert out.converged
    r1, r2 = residual(
        out.t1, out.t2, prepare_integrals(h, g, nocc=2), model=method.lower()
    )
    if method not in ("CCD", "LCCD"):
        assert np.max(np.abs(r1)) < 2e-9
    if method != "CCS":
        assert np.max(np.abs(r2)) < 2e-9
    if method == "CCS":
        assert abs(float(out.correlation_energy)) < 1e-12
    if method in ("CCD", "LCCD"):
        np.testing.assert_array_equal(out.t1, np.zeros((2, 2)))


def test_cc2_converged_energy_matches_pyscf(h4):
    from gradscf import cc
    from pyscf.cc import rccsd

    mf, h, g, _ = h4
    reference = (
        rccsd.RCCSD(mf)
        .set(cc2=True, conv_tol=1e-12, conv_tol_normt=1e-10, max_cycle=150)
        .run()
    )
    assert reference.converged
    result = cc.run_cc(h, g, nocc=2, config=cc.CCConfig(method="cc2"))
    np.testing.assert_allclose(result.correlation_energy, reference.e_corr, atol=2e-9)


@pytest.mark.parametrize("frozen", [None, 1, [0, 3]])
def test_ccsd_frozen_facade(h4, frozen):
    from gradscf import cc

    mf, h, g, _ = h4
    reference = mf.CCSD(frozen=frozen).set(conv_tol=1e-12, conv_tol_normt=1e-10).run()
    obj = cc.CCSD(cc.CCReference(h, g, 2, mf.mol.energy_nuc()), frozen=frozen).run()
    assert obj.converged
    np.testing.assert_allclose(obj.e_tot, reference.e_tot, atol=2e-9)
    e, t1, t2 = obj.kernel()
    assert t1.ndim == 2 and t2.ndim == 4 and e.shape == ()


def test_ccsd_and_triples_gradient(h4):
    from gradscf import cc
    from gradscf.cc.integrals import prepare_integrals

    mf, h, g, ref = h4
    v = prepare_integrals(h, g, nocc=2).fock - h

    def energy(t, triples):
        ht, gt = h - t * v, g * (1 + t)
        out = cc.run_cc(
            ht, gt, nocc=2, config=cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
        )
        return out.total_energy + (
            cc.triples_correction(ht, gt, out, nocc=2) if triples else 0.0
        )

    for triples in (False, True):
        fn = lambda t: energy(t, triples)
        ad = jax.jit(jax.grad(fn))(0.0)
        fd = (fn(1e-4) - fn(-1e-4)) / 2e-4
        np.testing.assert_allclose(ad, fd, atol=3e-7, rtol=3e-6)
        np.testing.assert_allclose(jax.jvp(fn, (0.0,), (1.0,))[1], ad, atol=1e-8)


def test_triples_matches_pyscf(h4):
    from gradscf import cc

    mf, h, g, ref = h4
    obj = cc.CCSD(cc.CCReference(h, g, 2, mf.mol.energy_nuc())).run()
    np.testing.assert_allclose(obj.ccsd_t(), ref.ccsd_t(), atol=1e-9)


def test_failure_and_unsupported_method(h4):
    from gradscf import cc

    _, h, g, _ = h4
    with pytest.raises(ValueError, match="method"):
        cc.CCConfig(method="ccsdt")
    cfg = cc.CCConfig(max_cycle=1, residual_tol=1e-13)
    fn = lambda t: cc.run_cc(h, g * (1 + t), nocc=2, config=cfg)
    assert not fn(0.0).converged
    assert not np.isfinite(jax.grad(lambda t: fn(t).total_energy)(0.0))


@pytest.mark.parametrize("model", ["ccsd", "ccd", "lccsd", "lccd", "ccs"])
def test_tiny_bch_oracle(h4, model):
    """Independent determinant-space exp(-T) H exp(T), not CC contractions."""
    from pyscf import fci
    from gradscf.cc.integrals import prepare_integrals
    from gradscf.cc.rccsd import residual, correlation_energy

    _, h, g, _ = h4
    n, no = 4, 2
    strings = fci.cistring.make_strings(range(n), no)
    dets = [int(a) | (int(b) << n) for a in strings for b in strings]
    lookup = {d: i for i, d in enumerate(dets)}
    dim = len(dets)

    def excitation(i, a, spin):
        op = np.zeros((dim, dim))
        i += spin * n
        a += spin * n
        for col, d in enumerate(dets):
            if not d & (1 << i) or d & (1 << a):
                continue
            phase = (-1) ** ((d & ((1 << i) - 1)).bit_count())
            out = d ^ (1 << i)
            phase *= (-1) ** ((out & ((1 << a) - 1)).bit_count())
            out |= 1 << a
            op[lookup[out], col] = phase
        return op

    e = {
        (i, a): excitation(i, a, 0) + excitation(i, a, 1)
        for i in range(no)
        for a in range(no, n)
    }
    rng = np.random.default_rng(21)
    t1 = 0.02 * rng.normal(size=(no, n - no))
    t2 = 0.03 * rng.normal(size=(no, no, n - no, n - no))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) * 0.5
    if model in ("ccd", "lccd"):
        t1[:] = 0
    if model == "ccs":
        t2[:] = 0
    t = sum(t1[i, a - no] * e[i, a] for i, a in e)
    t += sum(
        0.5 * t2[i, j, a - no, b - no] * e[i, a] @ e[j, b] for i, a in e for j, b in e
    )

    def exponential(x):
        total = np.eye(dim)
        term = np.eye(dim)
        for k in range(1, 2 * no + 1):
            term = term @ x / k
            total += term
        return total

    h2 = fci.direct_spin1.absorb_h1e(h, g, n, (no, no), 0.5)
    ham = np.column_stack(
        [
            fci.direct_spin1.contract_2e(h2, x.reshape(6, 6), n, (no, no)).ravel()
            for x in np.eye(dim)
        ]
    )
    ref = lookup[((1 << no) - 1) | (((1 << no) - 1) << n)]
    ket = np.eye(dim)[:, ref]
    transformed = (
        (ham + ham @ t - t @ ham) @ ket
        if model in ("lccd", "lccsd")
        else exponential(-t) @ ham @ exponential(t) @ ket
    )
    r1 = np.zeros_like(t1)
    r2 = np.zeros_like(t2)
    for i, a in e:
        r1[i, a - no] = (excitation(i, a, 0) @ ket) @ transformed
        for j, b in e:
            r2[i, j, a - no, b - no] = (
                excitation(i, a, 0) @ excitation(j, b, 1) @ ket
            ) @ transformed
    ints = prepare_integrals(h, g, nocc=no)
    if model in ("ccd", "lccd"):
        r1[:] = 0
    if model == "ccs":
        r2[:] = 0
    actual = residual(t1, t2, ints, model=model)
    np.testing.assert_allclose(actual[0], r1, atol=1e-11)
    np.testing.assert_allclose(actual[1], r2, atol=1e-11)
    np.testing.assert_allclose(
        correlation_energy(t1, t2, ints, model=model),
        transformed[ref] - ham[ref, ref],
        atol=1e-11,
    )


def test_water_ccsd_and_triples():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from gradscf import cc

    mol = pyscf.gto.M(
        atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g", verbose=0
    )
    mf = mol.RHF().run(conv_tol=1e-12)
    ref = mf.CCSD().set(conv_tol=1e-12, conv_tol_normt=1e-10).run()
    h = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), h.shape[0])
    ours = cc.CCSD(cc.CCReference(h, g, 5, mol.energy_nuc())).run()
    assert ours.converged
    np.testing.assert_allclose(ours.e_tot, ref.e_tot, atol=2e-9)
    np.testing.assert_allclose(ours.ccsd_t(), ref.ccsd_t(), atol=1e-9)


def test_noninteracting_fragments_are_additive(h4):
    from gradscf import cc

    _, h, g, _ = h4
    h2 = np.zeros((8, 8))
    g2 = np.zeros((8,) * 4)
    h2[:4, :4] = h
    h2[4:, 4:] = h
    g2[:4, :4, :4, :4] = g
    g2[4:, 4:, 4:, 4:] = g
    order = [0, 1, 4, 5, 2, 3, 6, 7]
    h2 = h2[np.ix_(order, order)]
    g2 = g2[np.ix_(order, order, order, order)]
    one = cc.run_cc(h, g, nocc=2)
    two = cc.run_cc(h2, g2, nocc=4)
    assert one.converged and two.converged
    np.testing.assert_allclose(two.total_energy, 2 * one.total_energy, atol=2e-9)


def test_gradscf_facade_two_electron_fci_limit():
    from gradscf import gto, dft, ci, cc

    mf = dft.RKS(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g"), xc="hf").run()
    ours = mf.CCSD().run()
    full = ci.CISD(mf, solver="dense").run()
    assert ours.converged
    np.testing.assert_allclose(ours.e_tot, full.e_tot, atol=2e-9)
    assert abs(float(ours.ccsd_t())) < 1e-12
    with pytest.raises(NotImplementedError, match="HF"):
        cc.CCSD(dft.RKS(mf.mol, xc="pbe")).run()


def test_all_frozen_returns_reference_and_rejects_bad_config(h4):
    from gradscf import cc

    mf, h, g, _ = h4
    out = cc.run_cc(h, g, nocc=2, frozen=2, nuclear_repulsion=mf.mol.energy_nuc())
    assert out.converged
    np.testing.assert_allclose(out.total_energy, mf.e_tot, atol=1e-10)
    for name in ["conv_tol", "residual_tol", "adjoint_tol", "denominator_tol"]:
        with pytest.raises(ValueError):
            cc.CCConfig(**{name: float("nan")})


def test_architecture_no_cc_solver_or_runtime_pyscf():
    import ast
    from pathlib import Path
    from gradscf import cc

    for path in Path(cc.__file__).parent.glob("*.py"):
        assert path.name not in ("solver.py", "solvers.py", "diis.py")
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(not x.name.startswith("pyscf") for x in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("pyscf")
                assert "ci" not in (node.module or "").split(".")


@pytest.mark.parametrize("method", ["ccs", "ccd", "cc2", "lccd", "lccsd"])
def test_variant_energy_response(h4, method):
    from gradscf import cc
    from gradscf.cc.integrals import prepare_integrals

    _, h, g, _ = h4
    v = prepare_integrals(h, g, nocc=2).fock - h
    cfg = cc.CCConfig(method=method, conv_tol=1e-12, residual_tol=1e-11)
    fn = lambda t: cc.run_cc(h - t * v, g * (1 + t), nocc=2, config=cfg).total_energy
    ad = jax.jit(jax.grad(fn))(0.0)
    fd = (fn(1e-4) - fn(-1e-4)) / 2e-4
    np.testing.assert_allclose(ad, fd, atol=3e-7, rtol=2e-6)


def test_amplitude_response_and_iteration_independence(h4):
    from gradscf import cc

    _, h, g, _ = h4

    def observable(t, damping, history):
        cfg = cc.CCConfig(
            damping=damping, diis_space=history, conv_tol=1e-12, residual_tol=1e-11
        )
        out = cc.run_cc(h, g * (1 + t), nocc=2, config=cfg)
        return jnp.sum(out.t2**2)

    base = lambda t: observable(t, 0.0, 6)
    other = lambda t: observable(t, 0.25, 4)
    ad = jax.jit(jax.grad(base))(0.0)
    fd = (base(1e-4) - base(-1e-4)) / 2e-4
    np.testing.assert_allclose(ad, fd, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(jax.jit(jax.grad(other))(0.0), ad, atol=1e-8)


def test_complex_initial_amplitudes_are_rejected(h4):
    from gradscf import cc

    _, h, g, _ = h4
    with pytest.raises(NotImplementedError, match="real"):
        cc.run_cc(h, g, nocc=2, t1=jnp.zeros((2, 2), dtype=complex))


def test_lower_precision_restart_amplitudes_match_integral_precision(h4):
    from gradscf import cc

    _, h, g, reference = h4
    out = cc.run_cc(h, g, nocc=2, t1=reference.t1.astype('float32'),
                    t2=reference.t2.astype('float32'))
    assert out.converged and out.t1.dtype == jnp.float64
    np.testing.assert_allclose(out.correlation_energy, reference.e_corr, atol=2e-9)
