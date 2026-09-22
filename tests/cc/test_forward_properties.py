"""Ground-state parity: UCC Lambda/(T) and restricted/unrestricted 1/2-RDMs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("frozen", [None, ([0], [])])
def test_unrestricted_lambda_and_densities(radical, frozen):
    from gradscf import cc
    mf, h, g = radical
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    result = cc.run_ucc(h, g, nocc=(2, 1), frozen=frozen, config=cfg)
    reference = mf.CCSD(frozen=frozen).set(conv_tol=1e-12, conv_tol_normt=1e-11).run()
    reference.solve_lambda()
    left = cc.solve_lambda(h, g, result, nocc=(2, 1), frozen=frozen, config=cfg)
    assert left.converged
    for a, b in zip(jax.tree.leaves((left.l1, left.l2)),
                    jax.tree.leaves((reference.l1, reference.l2))):
        np.testing.assert_allclose(a, b, atol=3e-8, rtol=0)
    dm1 = cc.make_rdm1(h, g, result, nocc=(2, 1), frozen=frozen, config=cfg)
    dm2 = cc.make_rdm2(h, g, result, nocc=(2, 1), frozen=frozen, config=cfg)
    for a, b in zip((*dm1, *dm2), (*reference.make_rdm1(), *reference.make_rdm2())):
        np.testing.assert_allclose(a, b, atol=3e-8, rtol=0)
    energy = sum(np.einsum('pq,qp', a, b) for a, b in zip(h, dm1))
    energy += sum(w*np.einsum('pqrs,pqrs', a, b) for w, a, b in zip((.5, 1., .5), g, dm2))
    np.testing.assert_allclose(energy, result.total_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose([np.trace(a) for a in dm1], (2, 1), atol=1e-12)


@pytest.mark.parametrize("frozen", [None, [0, 3]])
def test_restricted_rdm2_against_pyscf(frozen):
    from gradscf import cc
    from pyscf import gto, ao2mo
    mf = gto.M(atom="H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1",
                basis="sto-3g", verbose=0).RHF().run(conv_tol=1e-13)
    c = mf.mo_coeff
    h = c.T @ mf.get_hcore() @ c
    g = ao2mo.restore(1, ao2mo.kernel(mf.mol, c), 4)
    obj = cc.CCSD(cc.CCReference(h, g, 2), frozen=frozen,
                  conv_tol=1e-12, residual_tol=1e-11).run()
    reference = mf.CCSD(frozen=frozen).set(conv_tol=1e-12, conv_tol_normt=1e-11).run()
    dm1, dm2 = obj.make_rdm1(), obj.make_rdm2()
    np.testing.assert_allclose(dm2, reference.make_rdm2(), atol=3e-8, rtol=0)
    np.testing.assert_allclose(np.einsum('pqrr->pq', dm2), 3*dm1, atol=2e-9, rtol=0)
    energy = np.einsum('pq,qp', h, dm1)+.5*np.einsum('pqrs,pqrs', g, dm2)
    np.testing.assert_allclose(energy, obj.e_tot, atol=2e-10, rtol=0)


@pytest.fixture(scope="module")
def oh():
    from pyscf import gto, ao2mo
    from gradscf import cc
    mf = gto.M(atom="O 0 0 0; H 0 0 .97", basis="6-31g", spin=1,
                verbose=0).UHF().run(conv_tol=1e-13)
    ca, cb = mf.mo_coeff
    n = ca.shape[1]
    h = tuple(jnp.asarray(c.T @ mf.get_hcore() @ c) for c in (ca, cb))
    g = tuple(jnp.asarray(ao2mo.general(mf.mol, cs, compact=False).reshape((n,)*4))
              for cs in ((ca, ca, ca, ca), (ca, ca, cb, cb), (cb, cb, cb, cb)))
    return mf, h, g


@pytest.mark.parametrize("frozen", [None, 1])
def test_uccsd_t_oracle_and_facade(oh, frozen):
    from gradscf import cc
    mf, h, g = oh
    obj = cc.UCCSD(cc.UnrestrictedReference(h, g, (5, 4)), frozen=frozen,
                   conv_tol=1e-12, residual_tol=1e-10).run()
    reference = mf.CCSD(frozen=frozen).set(conv_tol=1e-12, conv_tol_normt=1e-11).run()
    expected = reference.ccsd_t()
    assert abs(expected) > 1e-7
    actual = obj.triples()
    assert actual.valid
    np.testing.assert_allclose(actual.energy, expected, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual.energy, actual.connected_component+actual.singles_component,
                               atol=1e-14, rtol=0)
    np.testing.assert_allclose(obj.ccsd_t(), expected, atol=2e-10, rtol=0)
    compiled = jax.jit(lambda h, g: cc.triples_correction(h, g, obj.result,
                                                        nocc=(5, 4), frozen=frozen))(h, g)
    np.testing.assert_allclose(compiled, expected, atol=2e-10, rtol=0)


def test_ucc_density_response_jit_and_failed_state(radical):
    from gradscf import cc
    _, h, g = radical
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    direction = jnp.diag(jnp.array([.13, -.07, .05]))
    def value(x):
        hs = (h[0]+x*direction, h[1])
        state = cc.run_ucc(hs, g, nocc=(2, 1), config=cfg)
        d1 = cc.make_rdm1(hs, g, state, nocc=(2, 1), config=cfg)
        d2 = cc.make_rdm2(hs, g, state, nocc=(2, 1), config=cfg)
        return jnp.sum(d1[0]*direction)+.17*jnp.sum(d2[1]**2)
    derivative = jax.jit(jax.grad(value))(0.)
    np.testing.assert_allclose(derivative, (value(1e-4)-value(-1e-4))/2e-4,
                               atol=2e-7, rtol=0)
    out = cc.run_ucc(h, g, nocc=(2, 1), config=cfg)._replace(converged=jnp.array(False))
    left = cc.solve_lambda(h, g, out, nocc=(2, 1), config=cfg)
    assert not left.converged
    assert all(np.isnan(a).all() for a in cc.make_rdm2(h, g, out, nocc=(2, 1), config=cfg))


def test_uccsd_t_closed_shell_limit_and_response():
    from gradscf import cc
    from pyscf import gto, ao2mo
    mf = gto.M(atom="H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1",
                basis="sto-3g", verbose=0).RHF().run(conv_tol=1e-13)
    c = mf.mo_coeff
    h = jnp.asarray(c.T @ mf.get_hcore() @ c)
    g = jnp.asarray(ao2mo.restore(1, ao2mo.kernel(mf.mol, c), 4))
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    restricted = cc.run_cc(h, g, nocc=2, config=cfg)
    unrestricted = cc.run_ucc((h, h), (g, g, g), nocc=(2, 2), config=cfg)
    et = cc.triples_correction((h, h), (g, g, g), unrestricted, nocc=(2, 2))
    assert abs(et) > 1e-7
    np.testing.assert_allclose(et, cc.triples_correction(h, g, restricted, nocc=2),
                               atol=2e-10, rtol=0)
    direction = jnp.diag(jnp.array([.2, -.1, .07, -.05]))
    def correction(x):
        hs = (h+x*direction, h)
        state = cc.run_ucc(hs, (g, g, g), nocc=(2, 2), config=cfg)
        return cc.triples_correction(hs, (g, g, g), state, nocc=(2, 2))
    actual = jax.jit(jax.grad(correction))(0.)
    np.testing.assert_allclose(actual, (correction(1e-4)-correction(-1e-4))/2e-4,
                               atol=1e-8, rtol=0)
    assert abs(actual) > 1e-8
    bad = unrestricted._replace(converged=jnp.array(False))
    assert not cc.evaluate_triples((h, h), (g, g, g), bad, nocc=(2, 2)).valid
    moved = h.at[0, 1].add(.01).at[1, 0].add(.01)
    assert not cc.evaluate_triples((moved, h), (g, g, g), unrestricted, nocc=(2, 2)).valid
    with pytest.raises(ValueError, match="max_virtual_triples"):
        cc.evaluate_triples((h, h), (g, g, g), unrestricted, nocc=(2, 2), max_virtual_triples=1)
    with pytest.raises(NotImplementedError, match="CCSD\\(T\\)"):
        cc.evaluate_triples((h, h), (g, g, g), unrestricted, nocc=(2, 2), variant="ccsd+t(ccsd)")


@pytest.mark.parametrize("nocc,frozen,method", [((1, 1), 1, "ccsd"),
    ((1, 0), None, "ccsd"), ((0, 1), None, "ccd")])
def test_empty_spin_or_frozen_density_and_triples(nocc, frozen, method):
    from gradscf import cc
    h = (jnp.diag(jnp.array([-1., .5])),)*2
    g = (jnp.zeros((2,)*4),)*3
    cfg = cc.CCConfig(method=method)
    state = cc.run_ucc(h, g, nocc=nocc, frozen=frozen, config=cfg)
    dm1 = cc.make_rdm1(h, g, state, nocc=nocc, frozen=frozen, config=cfg)
    dm2 = cc.make_rdm2(h, g, state, nocc=nocc, frozen=frozen, config=cfg)
    np.testing.assert_allclose([np.trace(a) for a in dm1], nocc, atol=1e-13)
    total2 = dm2[0]+dm2[1]+dm2[1].transpose(2, 3, 0, 1)+dm2[2]
    np.testing.assert_allclose(np.einsum('pqrr->pq', total2),
        (sum(nocc)-1)*(dm1[0]+dm1[1]), atol=1e-13)
    if method == "ccsd":
        out = cc.evaluate_triples(h, g, state, nocc=nocc, frozen=frozen)
        assert out.valid and out.energy == 0
        bad = state._replace(converged=jnp.array(False))
        assert not cc.evaluate_triples(h, g, bad, nocc=nocc, frozen=frozen).valid


def test_spin_density_parts_arbitrary_amplitudes_against_pyscf():
    """Exercise every spin-orbital contraction beyond the small radical oracle."""
    from types import SimpleNamespace
    from pyscf.cc import gccsd_rdm
    from gradscf.cc._spin_density import active_density_parts

    rng = np.random.default_rng(7)
    no, nv = 4, 5
    amplitudes = []
    for _ in range(2):
        singles = .1*rng.normal(size=(no, nv))
        doubles = .1*rng.normal(size=(no, no, nv, nv))
        doubles = .25*(doubles-doubles.swapaxes(0, 1)-doubles.swapaxes(2, 3)
                       +doubles.transpose(1, 0, 3, 2))
        amplitudes.extend((singles, doubles))
    actual1, actual2 = active_density_parts(*map(jnp.asarray, amplitudes))
    reference = SimpleNamespace(frozen=None)
    expected1 = gccsd_rdm.make_rdm1(reference, *amplitudes, with_mf=False)
    expected2 = gccsd_rdm.make_rdm2(reference, *amplitudes, with_dm1=False)
    np.testing.assert_allclose(actual1, expected1, atol=2e-13, rtol=0)
    np.testing.assert_allclose(actual2, expected2, atol=2e-13, rtol=0)
