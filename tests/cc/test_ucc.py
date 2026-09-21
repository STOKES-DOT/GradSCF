"""UCCSD oracle and implicit response tests; CPU float64, Hartree."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_uccsd_energy_amplitudes_and_jit(radical):
    from gradscf import cc
    mf, h, g = radical
    ref = mf.CCSD().set(conv_tol=1e-12, conv_tol_normt=1e-11).run()
    out = jax.jit(lambda h, g: cc.run_ucc(h, g, nocc=(2, 1),
                    nuclear_repulsion=mf.mol.energy_nuc(),
                    config=cc.CCConfig(conv_tol=1e-12, residual_tol=1e-10)))(h, g)
    assert out.converged
    np.testing.assert_allclose(out.total_energy, ref.e_tot, atol=2e-9)
    for actual, expected in zip((*out.t1, *out.t2), (*ref.t1, *ref.t2)):
        np.testing.assert_allclose(actual, expected, atol=2e-8)


def test_uccsd_energy_and_amplitude_response(radical):
    from gradscf import cc
    _, h, g = radical
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    d = jnp.diag(jnp.array([.2, -.1, .07]))
    def value(x):
        out = cc.run_ucc((h[0]+x*d, h[1]), (g[0], (1+.03*x)*g[1], g[2]), nocc=(2, 1), config=cfg)
        return out.total_energy + .13*jnp.sum(out.t1[0]**2) + .07*jnp.sum(out.t2[1]**2)
    actual = jax.jit(jax.grad(value))(0.)
    expected = (value(1e-4)-value(-1e-4))/2e-4
    np.testing.assert_allclose(actual, expected, atol=3e-8)


def test_ucc_facade_and_frozen(radical):
    from gradscf import cc
    from gradscf.scf.reference import UnrestrictedReference
    mf, h, g = radical
    obj = cc.UCCSD(UnrestrictedReference(h, g, (2, 1), mf.mol.energy_nuc()),
                   frozen=1, residual_tol=1e-11).run()
    ref = mf.CCSD(frozen=1).set(conv_tol=1e-12, conv_tol_normt=1e-10).run()
    assert obj.converged
    np.testing.assert_allclose(obj.e_tot, ref.e_tot, atol=2e-9)


def test_ucc_rejects_unimplemented_models(radical):
    from gradscf import cc
    _, h, g = radical
    with pytest.raises(NotImplementedError, match="CCSD.*CCD"):
        cc.run_ucc(h, g, nocc=(2, 1), config=cc.CCConfig(method="cc2"))


def test_oh_same_spin_amplitudes_and_noncanonical_residual():
    """OH/6-31G, R=0.97 Angstrom, doublet; both aa and bb blocks nonempty."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from pyscf.cc import gccsd
    from gradscf import cc
    from gradscf.cc.uccsd import prepare_ucc_integrals, residual
    from gradscf.cc.spin_amplitudes import SpinAmplitudeSpace
    mf = pyscf.gto.M(atom="O 0 0 0; H 0 0 .97", basis="6-31g", spin=1, verbose=0).UHF().run(conv_tol=1e-13)
    ca, cb = mf.mo_coeff
    n = ca.shape[1]
    h = tuple(c.T @ mf.get_hcore() @ c for c in (ca, cb))
    g = tuple(ao2mo.general(mf.mol, cs, compact=False).reshape((n,)*4)
              for cs in ((ca, ca, ca, ca), (ca, ca, cb, cb), (cb, cb, cb, cb)))
    ref = mf.CCSD().set(conv_tol=1e-12, conv_tol_normt=1e-10, max_cycle=150).run()
    assert ref.converged
    out = cc.run_ucc(h, g, nocc=(5, 4), nuclear_repulsion=mf.mol.energy_nuc(),
                     config=cc.CCConfig(residual_tol=1e-10, conv_tol=1e-12))
    assert out.converged
    np.testing.assert_allclose(out.total_energy, ref.e_tot, atol=2e-9)
    for a, b in zip((*out.t1, *out.t2), (*ref.t1, *ref.t2)):
        np.testing.assert_allclose(a, b, atol=3e-8)
    ints, no, nv = prepare_ucc_integrals(h, g, nocc=(5, 4))
    space = SpinAmplitudeSpace(no, nv)
    rng = np.random.default_rng(72)
    size = space.pack(*space.from_blocks(out.t1, out.t2, jnp.float64)).size
    t1, t2 = space.unpack(jnp.asarray(rng.normal(size=size)*.015))
    # Independent upstream residual using exactly the same spin ordering.
    eris = gccsd._PhysicistsERIs()
    for field in ints._fields:
        setattr(eris, field, np.asarray(getattr(ints, field)))
    # Include symmetric noncanonical occupied-virtual and same-spin block terms.
    spin = np.array([0]*5+[1]*4+[0]*nv[0]+[1]*nv[1])
    delta = rng.normal(size=eris.fock.shape)*.01
    delta = (delta+delta.T)*(spin[:, None] == spin[None, :])
    eris.fock = eris.fock+delta
    eris.mo_energy = np.diag(eris.fock)
    ints = ints._replace(fock=jnp.asarray(eris.fock))
    class Config:
        level_shift = 0.
    u1, u2 = gccsd.update_amps(Config(), np.asarray(t1), np.asarray(t2), eris)
    d = eris.mo_energy[:9, None]-eris.mo_energy[None, 9:]
    expected = ((u1-t1)*d, (u2-t2)*(d[:, None, :, None]+d[None, :, None, :]))
    for a, b in zip(residual(t1, t2, ints), expected):
        np.testing.assert_allclose(a, b, atol=3e-12)


def test_spin_swap_level_shift_and_restart(radical):
    from gradscf import cc
    _, h, g = radical
    base = cc.run_ucc(h, g, nocc=(2, 1), config=cc.CCConfig(residual_tol=1e-11))
    swapped = cc.run_ucc(h[::-1], (g[2], g[1].transpose(2, 3, 0, 1), g[0]),
                          nocc=(1, 2), config=cc.CCConfig(level_shift=.3, residual_tol=1e-11))
    assert base.converged and swapped.converged
    np.testing.assert_allclose(base.total_energy, swapped.total_energy, atol=2e-10)
    restarted = cc.run_ucc(h, g, nocc=(2, 1),
                           t1=tuple(t.astype(jnp.float32) for t in base.t1),
                           t2=tuple(t.astype(jnp.float32) for t in base.t2))
    assert restarted.converged and restarted.t1[0].dtype == jnp.float64
    np.testing.assert_allclose(restarted.total_energy, base.total_energy, atol=2e-9)


def test_unconverged_ucc_derivative_invalid(radical):
    from gradscf import cc
    _, h, g = radical
    def energy(x):
        return cc.run_ucc(h, tuple(x*a for a in g), nocc=(2, 1),
                          config=cc.CCConfig(max_cycle=1, residual_tol=1e-14)).total_energy
    assert not np.isfinite(jax.grad(energy)(1.))


@pytest.mark.parametrize("method", ["ccsd", "ccd"])
def test_all_frozen_empty_amplitude_response(method):
    from gradscf import cc
    h = (jnp.diag(jnp.array([-1., .5])),)*2
    g = (jnp.zeros((2,)*4),)*3
    def energy(x):
        return cc.run_ucc((x*h[0], h[1]), g, nocc=(1, 1), frozen=1,
                          config=cc.CCConfig(method=method)).total_energy
    np.testing.assert_allclose(energy(1.), -2., atol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(energy))(1.), -1., atol=1e-14)


def test_polarized_same_spin_doubles_against_fci():
    from gradscf import cc
    from pyscf.fci import direct_uhf
    rng = np.random.default_rng(61)
    h = np.diag([-1., -.7, .3, .5])
    b = rng.normal(size=(6, 4, 4))*.04
    b = b+b.transpose(0, 2, 1)
    g = np.einsum("Lpq,Lrs->pqrs", b, b)
    for nocc in ((2, 0), (0, 2)):
        out = cc.run_ucc((h, h), (g, g, g), nocc=nocc,
                         config=cc.CCConfig(residual_tol=1e-11, conv_tol=1e-12))
        expected, _ = direct_uhf.kernel((h, h), (g, g, g), 4, nocc, tol=1e-12)
        assert out.converged
        np.testing.assert_allclose(out.total_energy, expected, atol=2e-10)


@pytest.mark.parametrize("frozen", [None, ([0], [])])
def test_uccd_and_empty_spin_channels(radical, frozen):
    from gradscf import cc
    from gradscf.cc.uccsd import prepare_ucc_integrals, residual
    from gradscf.cc.spin_amplitudes import SpinAmplitudeSpace
    _, h, g = radical
    cfg = cc.CCConfig(method="ccd", residual_tol=1e-10)
    out = cc.run_ucc(h, g, nocc=(2, 1), frozen=frozen, config=cfg)
    assert out.converged
    ints, no, nv = prepare_ucc_integrals(h, g, nocc=(2, 1), frozen=frozen)
    space = SpinAmplitudeSpace(no, nv, "ccd")
    t = space.from_blocks(out.t1, out.t2, jnp.float64)
    assert max(float(jnp.max(jnp.abs(x), initial=0.)) for x in out.t1) == 0.
    assert jnp.max(jnp.abs(space.pack(*residual(*t, ints))), initial=0.) < 1e-9
    polarized = cc.run_ucc((jnp.diag(jnp.array([-1., .5])),)*2,
                          (jnp.zeros((2,)*4),)*3, nocc=(1, 0))
    assert polarized.converged and polarized.total_energy == -1.
    assert polarized.t1[1].shape == (0, 2)
