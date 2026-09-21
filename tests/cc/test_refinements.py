"""CC improvements with explicitly identified triples and response conventions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_ground import h4


def test_iterative_level_shift_preserves_energy_and_response(h4):
    from gradscf import cc

    _, h, g, _ = h4

    def solve(t, shift):
        return cc.run_cc(
            h,
            g * (1 + t),
            nocc=2,
            config=cc.CCConfig(level_shift=shift, residual_tol=1e-11, conv_tol=1e-12),
        )

    base = solve(0.0, 0.0)
    shifted = solve(0.0, 0.3)
    assert base.converged and shifted.converged
    np.testing.assert_allclose(shifted.total_energy, base.total_energy, atol=1e-10)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda t: solve(t, 0.3).total_energy))(0.0),
        jax.grad(lambda t: solve(t, 0.0).total_energy)(0.0),
        atol=1e-9,
    )
    with pytest.raises(ValueError):
        cc.CCConfig(level_shift=-0.1)


def test_triples_variants_and_diagnostics(h4):
    from gradscf import cc
    from pyscf.cc import ccsd_t_slow

    _, h, g, ref = h4
    result = cc.run_cc(h, g, nocc=2)
    urban = cc.evaluate_triples(h, g, result, nocc=2, variant="ccsd+t(ccsd)")
    conventional = cc.evaluate_triples(h, g, result, nocc=2)
    expected = ccsd_t_slow.kernel(
        ref, ref.ao2mo(), t1=np.zeros_like(ref.t1), t2=ref.t2, verbose=0
    )
    assert urban.valid and conventional.valid
    assert urban.singles_component == 0
    np.testing.assert_allclose(urban.energy, expected, atol=2e-10)
    np.testing.assert_allclose(conventional.energy, ref.ccsd_t(), atol=1e-9)
    np.testing.assert_allclose(
        conventional.energy,
        conventional.connected_component + conventional.singles_component,
        atol=1e-14,
    )
    assert conventional.min_abs_denominator > 0
    assert conventional.cc_residual_norm < 1e-8
    with pytest.raises(ValueError, match="variant"):
        cc.evaluate_triples(h, g, result, nocc=2, variant="ccsdt")


def test_triples_validate_state_even_when_space_is_empty(h4):
    from gradscf import cc

    _, h, g, _ = h4
    frozen = cc.run_cc(h, g, nocc=2, frozen=2)
    failed = frozen._replace(converged=jnp.asarray(False))
    assert np.isnan(cc.triples_correction(h, g, failed, nocc=2, frozen=2))
    result = cc.run_cc(h, g, nocc=2)
    changed = h.copy()
    changed[0, 0] += 0.1
    evaluated = cc.evaluate_triples(changed, g, result, nocc=2)
    assert not evaluated.valid and np.isnan(evaluated.energy)
    wrong = result._replace(t1=result.t1[:1])
    with pytest.raises(ValueError, match="shape"):
        cc.evaluate_triples(h, g, wrong, nocc=2)


@pytest.mark.parametrize("frozen", [None, 1, [0, 3]])
def test_ccsd_one_particle_density_matches_pyscf(h4, frozen):
    from gradscf import cc

    mf, h, g, _ = h4
    reference = mf.CCSD(frozen=frozen).set(conv_tol=1e-12, conv_tol_normt=1e-10).run()
    reference.solve_lambda()
    obj = cc.CCSD(cc.CCReference(h, g, 2), frozen=frozen).run()
    density = obj.make_rdm1()
    np.testing.assert_allclose(density, reference.make_rdm1(), atol=2e-7)
    np.testing.assert_allclose(density, density.T, atol=1e-12)
    np.testing.assert_allclose(jnp.trace(density), 4.0, atol=1e-10)


def test_density_response_matches_finite_difference(h4):
    from gradscf import cc

    _, h, g, _ = h4
    probe = jnp.asarray(np.random.default_rng(11).normal(size=h.shape))
    probe = (probe + probe.T) / 2
    cfg = cc.CCConfig(residual_tol=1e-11, conv_tol=1e-12)

    def value(t):
        gt = g * (1 + t)
        result = cc.run_cc(h, gt, nocc=2, config=cfg)
        return jnp.sum(cc.make_rdm1(h, gt, result, nocc=2, config=cfg) * probe)

    ad = jax.jit(jax.grad(value))(0.0)
    fd = (value(1e-4) - value(-1e-4)) / 2e-4
    np.testing.assert_allclose(ad, fd, atol=3e-7, rtol=3e-6)


@pytest.mark.parametrize("change", ["frozen", "method", "reference"])
def test_facade_rejects_stale_posthoc_state(h4, change):
    from gradscf import cc

    _, h, g, _ = h4
    source = cc.CCReference(h.copy(), g.copy(), 2)
    obj = cc.CCSD(source).run()
    if change == "frozen":
        obj.frozen = 1
    elif change == "method":
        obj.method = "cc2"
    else:
        source.h1[0, 0] += 0.1
    for call in (obj.ccsd_t, obj.solve_lambda, obj.make_rdm1):
        with pytest.raises(RuntimeError, match="kernel"):
            call()


def test_urban_triples_against_independent_determinant_moments(h4):
    """Urban correction is the squared WT2 triple moment over Fock gaps."""
    from pyscf import fci
    from gradscf import cc

    mf, h, g, _ = h4
    out = cc.run_cc(h, g, nocc=2)
    n, no = 4, 2
    strings = fci.cistring.make_strings(range(n), no)
    dets = [int(a) | (int(b) << n) for a in strings for b in strings]
    lookup = {d: i for i, d in enumerate(dets)}
    dim = len(dets)
    reference = ((1 << no) - 1) | (((1 << no) - 1) << n)
    ket = np.eye(dim)[:, lookup[reference]]

    def excitation(i, a, spin):
        i += spin * n
        a += spin * n
        op = np.zeros((dim, dim))
        for col, d in enumerate(dets):
            if not d & (1 << i) or d & (1 << a):
                continue
            sign = (-1) ** ((d & ((1 << i) - 1)).bit_count())
            v = d ^ (1 << i)
            sign *= (-1) ** ((v & ((1 << a) - 1)).bit_count())
            v |= 1 << a
            op[lookup[v], col] = sign
        return op

    operators = {
        (i, a): excitation(i, a, 0) + excitation(i, a, 1)
        for i in range(no)
        for a in range(no, n)
    }
    t2ket = sum(
        0.5
        * float(out.t2[i, j, a - no, b - no])
        * (operators[i, a] @ operators[j, b] @ ket)
        for i, a in operators
        for j, b in operators
    )
    h2 = fci.direct_spin1.absorb_h1e(h, g, n, (no, no), 0.5)
    moment = fci.direct_spin1.contract_2e(h2, t2ket.reshape(6, 6), n, (no, no)).ravel()
    eps = np.tile(mf.mo_energy, 2)
    reference_level = sum(eps[i] for i in range(2 * n) if reference & (1 << i))
    expected = 0.0
    for row, det in enumerate(dets):
        if (reference & ~det).bit_count() != 3:
            continue
        level = sum(eps[i] for i in range(2 * n) if det & (1 << i))
        expected += moment[row] ** 2 / (reference_level - level)
    value = cc.triples_correction(h, g, out, nocc=2, variant="ccsd+t(ccsd)")
    np.testing.assert_allclose(value, expected, atol=2e-11)


def test_urban_triples_derivative_includes_amplitude_response(h4):
    from gradscf import cc
    from gradscf.cc.integrals import prepare_integrals

    _, h, g, _ = h4
    potential = prepare_integrals(h, g, nocc=2).fock - h
    cfg = cc.CCConfig(residual_tol=1e-11, conv_tol=1e-12)

    def correction(t):
        ht, gt = h - t * potential, g * (1 + t)
        result = cc.run_cc(ht, gt, nocc=2, config=cfg)
        return cc.triples_correction(ht, gt, result, nocc=2, variant="ccsd+t(ccsd)")

    ad = jax.jit(jax.grad(correction))(0.0)
    fd = (correction(1e-4) - correction(-1e-4)) / 2e-4
    np.testing.assert_allclose(ad, fd, atol=2e-9, rtol=2e-6)
