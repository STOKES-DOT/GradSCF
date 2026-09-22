"""Restricted QCISD/(T) against PySCF's separate optimized QCI implementation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def h4():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    mf = pyscf.gto.M(atom="H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1",
                     basis="sto-3g", verbose=0).RHF().run(conv_tol=1e-13)
    c = mf.mo_coeff
    return mf, jnp.asarray(c.T @ mf.get_hcore() @ c), jnp.asarray(
        ao2mo.restore(1, ao2mo.kernel(mf.mol, c), 4))


@pytest.mark.parametrize("noncanonical", [False, True])
def test_qcisd_random_residual_and_energy(h4, noncanonical):
    from gradscf.cc.integrals import prepare_integrals
    from gradscf.cc.rccsd import residual, correlation_energy
    from pyscf.cc import qcisd
    mf, h, g = h4
    reference = qcisd.QCISD(mf)
    eris = reference.ao2mo()
    rng = np.random.default_rng(67)
    if noncanonical:
        perturb = rng.normal(size=(4, 4))*.03
        perturb = (perturb+perturb.T)/2
        h = h+perturb
        eris.fock = eris.fock+perturb
        eris.mo_energy = np.diag(eris.fock).copy()
    t1 = rng.normal(size=(2, 2))*.04
    t2 = rng.normal(size=(2, 2, 2, 2))*.05
    t2 = (t2+t2.transpose(1, 0, 3, 2))/2
    updated = reference.update_amps(t1, t2, eris)
    d1 = eris.mo_energy[:2, None]-eris.mo_energy[None, 2:]
    expected = ((updated[0]-t1)*d1,
                (updated[1]-t2)*(d1[:, None, :, None]+d1[None, :, None, :]))
    ints = prepare_integrals(h, g, nocc=2)
    actual = jax.jit(lambda a, b: residual(a, b, ints, model="qcisd"))(t1, t2)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=0)
    np.testing.assert_allclose(correlation_energy(t1, t2, ints, model="qcisd"),
                               reference.energy(t1, t2, eris), atol=2e-12, rtol=0)


@pytest.mark.parametrize("frozen", [None, 1, [0, 3]])
def test_qcisd_energy_amplitudes_and_triples(h4, frozen):
    from gradscf import cc
    from pyscf.cc import qcisd, ccsd_t_slow, qcisd_t_slow
    mf, h, g = h4
    ref = qcisd.QCISD(mf, frozen=frozen).set(conv_tol=1e-12, conv_tol_normt=1e-11,
                                           max_cycle=150).run()
    assert ref.converged
    obj = cc.QCISD(cc.CCReference(h, g, 2, mf.mol.energy_nuc()), frozen=frozen,
                   conv_tol=1e-12, residual_tol=1e-11).run()
    assert obj.converged
    np.testing.assert_allclose(obj.e_tot, ref.e_tot, atol=2e-9, rtol=0)
    np.testing.assert_allclose(obj.t1, ref.t1, atol=2e-8, rtol=0)
    np.testing.assert_allclose(obj.t2, ref.t2, atol=2e-8, rtol=0)
    correction = obj.triples()
    assert correction.valid
    np.testing.assert_allclose(correction.energy, ref.qcisd_t(), atol=2e-10, rtol=0)
    np.testing.assert_allclose(obj.qcisd_t(), correction.energy, atol=1e-14, rtol=0)
    if frozen is None:
        eris = ref.ao2mo()
        actual_t1, actual_t2 = np.asarray(obj.t1), np.asarray(obj.t2)
        connected = qcisd_t_slow.kernel(ref, eris, np.zeros_like(actual_t1), actual_t2)
        wrong_method = ccsd_t_slow.kernel(ref, eris, actual_t1, actual_t2)
        np.testing.assert_allclose(correction.connected_component, connected, atol=1e-12)
        np.testing.assert_allclose(correction.energy, 2*wrong_method-connected, atol=1e-12)
        assert abs(correction.energy-wrong_method) > 1e-12


def test_qcisd_total_and_amplitude_response(h4):
    from gradscf import cc
    _, h, g = h4
    cfg = cc.CCConfig(method="qcisd", conv_tol=1e-12, residual_tol=1e-11)
    direction = jnp.diag(jnp.array([.12, -.04, .07, -.03]))
    def value(x):
        hs = h+x*direction
        out = cc.run_cc(hs, g, nocc=2, config=cfg)
        return (out.total_energy+cc.triples_correction(hs, g, out, nocc=2, variant="qcisd(t)")
                +.13*jnp.sum(out.t1**2)+.09*jnp.sum(out.t2**2))
    actual = jax.jit(jax.grad(value))(0.)
    np.testing.assert_allclose(actual, (value(1e-4)-value(-1e-4))/2e-4, atol=2e-7, rtol=0)


def test_qcisd_model_density_and_adjoint(h4):
    from gradscf import cc
    _, h, g = h4
    cfg = cc.CCConfig(method="qcisd", conv_tol=1e-12, residual_tol=1e-11)
    state = cc.run_cc(h, g, nocc=2, config=cfg)
    left = cc.solve_lambda(h, g, state, nocc=2, config=cfg)
    assert left.converged
    dm1 = cc.make_rdm1(h, g, state, nocc=2, config=cfg)
    direction = jnp.array([[.1, .03, .02, .01], [.03, -.1, .04, .05],
                          [.02, .04, .08, -.02], [.01, .05, -.02, .09]])
    value = lambda x: cc.run_cc(h+x*direction, g, nocc=2, config=cfg).total_energy
    np.testing.assert_allclose(jnp.sum(dm1*direction), (value(1e-4)-value(-1e-4))/2e-4,
                               atol=2e-7, rtol=0)
    np.testing.assert_allclose(jnp.trace(dm1), 4., atol=1e-11)
    with pytest.raises(NotImplementedError, match="2-RDM"):
        cc.make_rdm2(h, g, state, nocc=2, config=cfg)


def test_qcisd_state_and_scope_guards(h4):
    from gradscf import cc
    _, h, g = h4
    qci = cc.QCISD(cc.CCReference(h, g, 2)).run()
    ccsd = cc.CCSD(cc.CCReference(h, g, 2)).run()
    assert not cc.evaluate_triples(h, g, ccsd.result, nocc=2, variant="qcisd(t)").valid
    assert not cc.evaluate_triples(h, g, qci.result, nocc=2).valid
    with pytest.raises(ValueError, match="amplitudes"):
        qci.ccsd_t()
    with pytest.raises(ValueError, match="amplitudes"):
        ccsd.qcisd_t()
    with pytest.raises(NotImplementedError, match="restricted"):
        cc.QCISD(cc.UnrestrictedReference((h, h), (g, g, g), (2, 2)))
    with pytest.raises(NotImplementedError, match="CCSD"):
        qci.triples(orbital_basis="semicanonical")
    empty = cc.QCISD(cc.CCReference(h, g, 2), frozen=2).run()
    assert empty.qcisd_t() == 0
    bad = empty.result._replace(converged=jnp.array(False))
    assert not cc.evaluate_triples(h, g, bad, nocc=2, frozen=2, variant="qcisd(t)").valid
    from gradscf.cc.ground import METHODS
    wrong = empty.result._replace(method_id=jnp.asarray(METHODS.index("ccsd")))
    assert not cc.evaluate_triples(h, g, wrong, nocc=2, frozen=2, variant="qcisd(t)").valid
    qci.frozen = 1
    with pytest.raises(RuntimeError, match="changed"):
        qci.qcisd_t()


def test_native_qcisd_facade():
    from gradscf import gto, dft
    mf = dft.RKS(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g"), xc="hf", conv_tol=1e-12).run()
    qci = mf.QCISD().run()
    assert qci.converged and qci.qcisd_t() == 0


def test_qcisd_residual_is_quadratic(h4):
    from gradscf.cc.rccsd import residual
    from gradscf.cc.integrals import prepare_integrals
    _, h, g = h4
    ints = prepare_integrals(h, g, nocc=2)
    rng = np.random.default_rng(68)
    t1 = jnp.asarray(rng.normal(size=(2, 2))*.12)
    t2 = jnp.asarray(rng.normal(size=(2, 2, 2, 2))*.09)
    t2 = (t2+t2.transpose(1, 0, 3, 2))/2
    def third_difference(model):
        values = [residual(k*t1, k*t2, ints, model=model) for k in range(4)]
        return jax.tree.map(lambda a, b, c, d: d-3*c+3*b-a, *values)
    for value in third_difference("qcisd"):
        np.testing.assert_allclose(value, 0., atol=3e-14, rtol=0)
    assert max(np.max(np.abs(v)) for v in third_difference("ccsd")) > 1e-5


def test_qcisd_noninteracting_fragment_additivity(h4):
    from gradscf import cc
    _, h, g = h4
    cfg = cc.CCConfig(method="qcisd", conv_tol=1e-12, residual_tol=1e-11)
    fragment = cc.run_cc(h, g, nocc=2, config=cfg)
    h2, g2 = np.zeros((8, 8)), np.zeros((8,)*4)
    for start in (0, 4):
        s = slice(start, start+4)
        h2[s, s], g2[s, s, s, s] = h, g
    order = [0, 1, 4, 5, 2, 3, 6, 7]
    out = cc.run_cc(h2[np.ix_(order, order)], g2[np.ix_(order, order, order, order)],
                    nocc=4, config=cfg)
    assert fragment.converged and out.converged
    np.testing.assert_allclose(out.total_energy, 2*fragment.total_energy, atol=2e-9, rtol=0)


def test_water_qcisd_t_against_pyscf():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from gradscf import cc
    mf = pyscf.gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g",
                     verbose=0).RHF().run(conv_tol=1e-13)
    c = mf.mo_coeff
    h = c.T @ mf.get_hcore() @ c
    g = ao2mo.restore(1, ao2mo.kernel(mf.mol, c), c.shape[1])
    reference = pyscf.cc.QCISD(mf).set(conv_tol=1e-12, conv_tol_normt=1e-11, max_cycle=150).run()
    actual = cc.QCISD(cc.CCReference(h, g, 5, mf.mol.energy_nuc()),
                      conv_tol=1e-12, residual_tol=1e-11).run()
    assert actual.converged and reference.converged
    np.testing.assert_allclose(actual.e_tot, reference.e_tot, atol=2e-9, rtol=0)
    np.testing.assert_allclose(actual.qcisd_t(), reference.qcisd_t(), atol=2e-10, rtol=0)
    assert abs(actual.qcisd_t()) > 1e-7


def test_qcisd_unconverged_derivative_is_invalid(h4):
    from gradscf import cc
    _, h, g = h4
    def energy(x):
        return cc.run_cc(h, x*g, nocc=2, config=cc.CCConfig(method="qcisd", max_cycle=1,
                                                           residual_tol=1e-14)).total_energy
    assert not np.isfinite(jax.jit(jax.grad(energy))(1.))
