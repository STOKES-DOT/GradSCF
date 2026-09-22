"""Full static BSE: independent doubled oracle, TDHF limit and optical AD."""

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def model():
    with np.load(Path(__file__).parent / "data/quack_static_seed83.npz") as data:
        qp, e, l = [
            jnp.asarray(data[k]) for k in ("qp_energy", "screening_energy", "factors")
        ]
    rng = np.random.default_rng(21)
    d = rng.normal(size=(3, 5, 5)) * 0.2
    return qp, e, l, jnp.asarray((d + d.transpose(0, 2, 1)) * 0.5)


def independent_blocks(qp, e, l, singlet):
    l = np.asarray(l)
    eps = np.eye(l.shape[0])
    pairs = [(i, a) for i in range(2) for a in range(2, 5)]
    for i, a in pairs:
        eps += 4 * np.outer(l[:, i, a], l[:, i, a]) / (e[a] - e[i])
    a, b = np.zeros((6, 6)), np.zeros((6, 6))
    for p, (i, v) in enumerate(pairs):
        for q, (j, w) in enumerate(pairs):
            bare = (2 if singlet else 0) * np.dot(l[:, i, v], l[:, j, w])
            a[p, q] = bare - np.dot(l[:, i, j], np.linalg.solve(eps, l[:, v, w]))
            b[p, q] = bare - np.dot(l[:, i, w], np.linalg.solve(eps, l[:, v, j]))
        a[p, p] += qp[v] - qp[i]
    return a, b


@pytest.mark.parametrize("singlet", [True, False])
def test_full_bse_blocks_roots_and_optics(singlet):
    from gradscf import bse
    from gradscf.gw.screened import build_static_screening

    qp, e, l, d = model()
    space = bse.make_bse_space(5, 2)
    state = build_static_screening(e, l, occupied=space.occupied, virtual=space.virtual)
    a, b = independent_blocks(qp, e, l, singlet)
    ao, bo = bse.build_bse_operators(qp, l, space, state, singlet=singlet, block_size=2)
    for op, expected in ((ao, a), (bo, b)):
        np.testing.assert_allclose(op.apply(jnp.eye(6)), expected, atol=2e-13)
        np.testing.assert_allclose(op.diagonal, np.diag(expected), atol=2e-13)
    cfg = bse.BSEConfig(
        tda=False, solver="dense", nroots=3, singlet=singlet, conv_tol=1e-11
    )
    out = jax.jit(lambda q, e, l: bse.run_bse(q, e, l, space, config=cfg))(qp, e, l)
    expected, z = np.linalg.eig(np.block([[a, b], [-b, -a]]))
    order = np.argsort(expected.real)[6:9]
    z = z[:, order].real
    z /= np.sqrt(np.sum(z[:6] ** 2 - z[6:] ** 2, axis=0))
    mu = np.sqrt(2) * np.einsum(
        "xk,ks->sx", np.asarray(d)[:, :2, 2:].reshape(3, 6), z[:6] + z[6:]
    )
    f = 2 / 3 * expected[order].real * np.sum(mu**2, axis=1) if singlet else np.zeros(3)
    np.testing.assert_allclose(out.excitation_energies, expected[order], atol=2e-12)
    np.testing.assert_allclose(bse.oscillator_strengths(out, d, space), f, atol=2e-12)
    assert np.all(out.converged & out.response_valid) and np.all(out.stable)
    assert np.all(out.stability_margins > 0)
    assert np.linalg.norm(out.y_amplitudes) > 1e-4


def test_full_bse_joint_first_order_response():
    from gradscf import bse

    qp, e, l, d = model()
    space = bse.make_bse_space(5, 2)
    cfg = bse.BSEConfig(tda=False, solver="dense", nroots=2)

    def loss(t):
        out = bse.run_bse(
            qp * (1 + 0.03 * t),
            e * (1 - 0.05 * t),
            l * (1 + 0.2 * t),
            space,
            config=cfg,
        )
        return jnp.sum(
            out.excitation_energies * jnp.array([0.3, 0.8])
            + bse.oscillator_strengths(out, d * (1 + 0.1 * t), space)
        )

    grad = jax.jit(jax.grad(loss))(0.0)
    fd = (loss(1e-5) - loss(-1e-5)) / 2e-5
    np.testing.assert_allclose(grad, fd, atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(jax.jvp(loss, (0.0,), (1.0,))[1], grad, atol=1e-11)


def test_full_bse_eager_api_and_capacity():
    from gradscf import bse

    qp, e, l, d = model()
    ref = bse.BSEReference(qp, e, l, 2, dipole_mo=d)
    calc = bse.BSE(ref, tda=False, solver="dense", nroots=2).run()
    assert np.all(np.isfinite(calc.oscillator_strength()))
    with pytest.raises(ValueError, match="max_dense"):
        bse.BSE(ref, tda=False, solver="dense", nroots=2, max_dense=2).run()
    with pytest.raises(NotImplementedError, match="dense"):
        bse.BSEConfig(tda=False, solver="davidson")
    calc.tda = True
    with pytest.raises(RuntimeError, match="changed"):
        calc.oscillator_strength()


def test_full_bse_energy_only_property_derivative_is_invalid():
    from gradscf import bse

    qp, e, l, d = model()
    space = bse.make_bse_space(5, 2)
    cfg = bse.BSEConfig(
        tda=False, solver="dense", nroots=2, gradient_mode="eigenvalue_only"
    )

    def loss(t):
        out = bse.run_bse(qp * t, e, l, space, config=cfg)
        return bse.oscillator_strengths(out, d, space).sum()

    assert np.isfinite(loss(1.0))
    assert not np.isfinite(jax.grad(loss)(1.0))


@pytest.mark.parametrize("singlet", [True, False])
def test_full_bare_screening_hf_limit_matches_pyscf_tdhf(singlet):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from gradscf import bse

    mf = (
        pyscf.gto.M(
            atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g", verbose=0
        )
        .RHF()
        .run(conv_tol=1e-13)
    )
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff), n)
    values, vectors = np.linalg.eigh(eri.reshape(n * n, n * n))
    keep = values > 1e-12
    l = (vectors[:, keep] * np.sqrt(values[keep])).T.reshape(-1, n, n)
    l = (l + l.transpose(0, 2, 1)) * 0.5
    space = bse.make_bse_space(n, mf.mol.nelectron // 2)
    unscreened = bse.make_bse_space(n, space.nocc, occupied=(), virtual=())
    out = bse.run_bse(
        mf.mo_energy,
        mf.mo_energy,
        jnp.asarray(l),
        space,
        screening_space=unscreened,
        config=bse.BSEConfig(tda=False, solver="dense", nroots=3, singlet=singlet),
    )
    td = mf.TDHF().set(nstates=3, singlet=singlet, conv_tol=1e-11).run()
    np.testing.assert_allclose(out.excitation_energies, td.e, atol=2e-9, rtol=0)
    d = np.einsum(
        "xmn,mi,nj->xij", mf.mol.intor("int1e_r", comp=3), mf.mo_coeff, mf.mo_coeff
    )
    expected = td.oscillator_strength() if singlet else np.zeros(3)
    np.testing.assert_allclose(
        bse.oscillator_strengths(out, d, space), expected, atol=2e-9, rtol=0
    )


def test_full_independent_limit_and_unstable_optics():
    from gradscf import bse

    space = bse.make_bse_space(2, 1)
    e = jnp.array([-1.0, 1.0])
    cfg = bse.BSEConfig(tda=False, solver="dense", nroots=1)
    result = bse.run_bse(e, e, jnp.zeros((0, 2, 2)), space, config=cfg)
    np.testing.assert_allclose(result.excitation_energies, [2.0], atol=1e-14)
    np.testing.assert_allclose(result.y_amplitudes, 0, atol=1e-14)
    factors = jnp.array([[[2.0, 0.0], [0.0, 2.0]]])
    result = bse.run_bse(e, e, factors, space, config=cfg)
    assert not np.any(result.stable | result.converged)
    np.testing.assert_allclose(result.stability_margins, [-2.0, -2.0], atol=1e-14)
    assert np.isnan(bse.oscillator_strengths(result, jnp.ones((3, 2, 2)), space)).all()
