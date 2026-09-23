"""Closed-shell EE/IP/EA-CCSD actions, spectra and full CC amplitude response."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def reference():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from gradscf import cc

    mol = pyscf.gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", verbose=0)
    mf = mol.RHF().run(conv_tol=1e-13)
    expected = mf.CCSD().run(conv_tol=1e-13, conv_tol_normt=1e-12)
    h = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 2)
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    ground = cc.run_cc(h, g, nocc=1, config=cfg)
    assert ground.converged
    return h, g, ground, cfg, expected


@pytest.mark.parametrize("sector", ["ee", "ip", "ea"])
def test_sector_actions_and_roots_match_pyscf(reference, sector):
    from gradscf.cc.eom import EOMConfig, run_eom, build_eom_operator
    from pyscf.cc import eom_rccsd

    h, g, ground, cfg, expected = reference
    cls = {"ee": eom_rccsd.EOMEESinglet, "ip": eom_rccsd.EOMIP, "ea": eom_rccsd.EOMEA}[
        sector
    ]
    oracle = cls(expected)
    imds = oracle.make_imds()
    config = EOMConfig(sector=sector, nroots=2, conv_tol=1e-10)
    op, space, valid = build_eom_operator(
        h, g, ground, nocc=1, config=config, cc_config=cfg
    )
    assert valid
    columns = []
    for vector in np.eye(space.size):
        r1, r2 = space.unpack(vector)
        pyvector = oracle.amplitudes_to_vector(np.asarray(r1), np.asarray(r2))
        pyoutput = oracle.matvec(pyvector, imds)
        s1, s2 = oracle.vector_to_amplitudes(pyoutput)
        columns.append(np.asarray(space.pack(s1, s2)))
    matrix = np.stack(columns, axis=1)
    np.testing.assert_allclose(
        op.apply(jnp.eye(space.size)), matrix, atol=2e-10, rtol=0
    )
    np.testing.assert_allclose(
        op.T.apply(jnp.eye(space.size)), matrix.T, atol=2e-10, rtol=0
    )
    result = jax.jit(
        lambda h, g: run_eom(h, g, ground, nocc=1, config=config, cc_config=cfg)
    )(h, g)
    np.testing.assert_allclose(
        result.energies, np.sort(np.linalg.eigvals(matrix).real)[:2], atol=2e-10, rtol=0
    )
    assert np.all(result.converged & result.response_valid)
    np.testing.assert_allclose(
        result.left_vectors.T @ result.right_vectors, np.eye(2), atol=1e-11
    )


@pytest.mark.parametrize("sector", ["ee", "ip", "ea"])
def test_full_cc_to_eom_energy_derivative(reference, sector):
    from gradscf import cc
    from gradscf.cc.eom import EOMConfig, run_eom

    h, g, _, cfg, _ = reference
    config = EOMConfig(sector=sector, nroots=1, conv_tol=1e-10)
    direction = jnp.array([[0.2, 0.04], [0.04, -0.1]])

    def energy(t):
        hh = h + t * direction
        gg = g * (1 + 0.02 * t)
        ground = cc.run_cc(hh, gg, nocc=1, config=cfg)
        return run_eom(hh, gg, ground, nocc=1, config=config, cc_config=cfg).energies[0]

    ad = jax.jit(jax.grad(energy))(0.0)
    np.testing.assert_allclose(
        ad, (energy(1e-4) - energy(-1e-4)) / 2e-4, atol=2e-7, rtol=2e-6
    )
    np.testing.assert_allclose(jax.jvp(energy, (0.0,), (1.0,))[1], ad, atol=1e-9)


def test_eom_rejects_wrong_ground_model_and_capacity(reference):
    from gradscf.cc.eom import EOMConfig, run_eom

    h, g, ground, cfg, _ = reference
    for bad in [
        ground._replace(converged=False),
        ground._replace(method_id=jnp.asarray(0)),
    ]:
        out = run_eom(h, g, bad, nocc=1, config=EOMConfig(nroots=1), cc_config=cfg)
        assert not np.any(out.converged | out.response_valid)
    with pytest.raises(ValueError, match="max_dense"):
        run_eom(
            h, g, ground, nocc=1, config=EOMConfig(nroots=1, max_dense=1), cc_config=cfg
        )

    empty = ground._replace(t1=jnp.zeros((1, 0)), t2=jnp.zeros((1, 1, 0, 0)))
    with pytest.raises(ValueError, match="empty EOM sector"):
        run_eom(h, g, empty, nocc=1)
    with pytest.raises(ValueError, match="max_intermediate_elements"):
        run_eom(h, g, ground, nocc=1, config=EOMConfig(max_intermediate_elements=1))


@pytest.mark.parametrize("sector", ["ee", "ip", "ea"])
@pytest.mark.parametrize("frozen", [None, [0, 5]])
def test_larger_sector_and_frozen_actions(sector, frozen):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from pyscf.cc import eom_rccsd
    from gradscf import cc
    from gradscf.cc.eom import EOMConfig, build_eom_operator, run_eom

    atom = (
        "H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1"
        if frozen is None
        else "Li 0 0 0; H 0 0 1.6"
    )
    mf = pyscf.gto.M(atom=atom, basis="sto-3g", verbose=0).RHF().run(conv_tol=1e-13)
    pycc = mf.CCSD(frozen=frozen).run(conv_tol=1e-13, conv_tol_normt=1e-12)
    n = mf.mo_coeff.shape[1]
    no = mf.mol.nelectron // 2
    h = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff), n)
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11, max_cycle=200)
    ground = cc.run_cc(h, g, nocc=no, frozen=frozen, config=cfg)
    assert ground.converged
    oracle = {
        "ee": eom_rccsd.EOMEESinglet,
        "ip": eom_rccsd.EOMIP,
        "ea": eom_rccsd.EOMEA,
    }[sector](pycc)
    imds = oracle.make_imds()
    config = EOMConfig(sector=sector, nroots=2)
    op, space, valid = build_eom_operator(
        h, g, ground, nocc=no, frozen=frozen, config=config, cc_config=cfg
    )
    matrix = []
    for v in np.eye(space.size):
        r1, r2 = space.unpack(v)
        output = oracle.matvec(
            oracle.amplitudes_to_vector(np.asarray(r1), np.asarray(r2)), imds
        )
        matrix.append(np.asarray(space.pack(*oracle.vector_to_amplitudes(output))))
    matrix = np.stack(matrix, axis=1)
    np.testing.assert_allclose(op.apply(jnp.eye(space.size)), matrix, atol=5e-9, rtol=0)
    for solver in ("dense", "davidson"):
        result = run_eom(
            h,
            g,
            ground,
            nocc=no,
            frozen=frozen,
            config=EOMConfig(sector=sector, nroots=2, solver=solver),
            cc_config=cfg,
        )
        assert np.all(result.converged)
        np.testing.assert_allclose(
            result.energies,
            np.sort(np.linalg.eigvals(matrix).real)[:2],
            atol=5e-9,
            rtol=0,
        )


def test_eager_facade_and_stale_snapshot(reference):
    from gradscf import cc

    h, g, _, _, _ = reference
    obj = cc.CCSD(cc.CCReference(h, g, 1), conv_tol=1e-12, residual_tol=1e-11).run()
    response = cc.EOMEE(obj, nroots=1).run()
    expected = response.e.copy()
    e, v = obj.eomee_ccsd_singlet(nroots=1)
    np.testing.assert_allclose(e, expected[0], atol=1e-12)
    assert v.ndim == 1
    assert cc.EOMIP(obj, nroots=1).run().converged.all()
    assert cc.EOMEA(obj, nroots=1).run().converged.all()
    obj.frozen = [0]
    with pytest.raises(RuntimeError, match="changed"):
        response.check_source()
    with pytest.raises(RuntimeError, match="changed"):
        response.kernel()


def test_noninteracting_sector_signs_and_exact_h2_particle_numbers(reference):
    from gradscf import cc
    from gradscf.cc.eom import EOMConfig, run_eom
    from pyscf import fci

    h, g, ground, cfg, _ = reference
    e0 = fci.direct_spin1.kernel(h, g, 2, (1, 1))[0]
    ip = fci.direct_spin1.kernel(h, g, 2, (1, 0))[0] - e0
    ea = fci.direct_spin1.kernel(h, g, 2, (2, 1))[0] - e0
    for sector, expected in [("ip", ip), ("ea", ea)]:
        actual = run_eom(
            h, g, ground, nocc=1, config=EOMConfig(sector=sector), cc_config=cfg
        )
        np.testing.assert_allclose(actual.energies[0], expected, atol=2e-10, rtol=0)
    h0 = jnp.diag(jnp.array([-1.0, -0.2]))
    g0 = jnp.zeros((2, 2, 2, 2))
    ground0 = cc.run_cc(h0, g0, nocc=1)
    for sector, expected in [("ee", 0.8), ("ip", 1.0), ("ea", -0.2)]:
        actual = run_eom(h0, g0, ground0, nocc=1, config=EOMConfig(sector=sector))
        np.testing.assert_allclose(actual.energies[0], expected, atol=1e-12)
        assert actual.response_valid[0]


def test_stale_and_unconverged_cc_energy_response_is_invalid(reference):
    from gradscf.cc.eom import EOMConfig, run_eom

    h, g, ground, cfg, _ = reference
    bad = ground._replace(converged=False)
    f = lambda t: run_eom(
        h, g * t, bad, nocc=1, config=EOMConfig(), cc_config=cfg
    ).energies[0]
    assert not np.isfinite(jax.grad(f)(1.0))
    mismatch = run_eom(h, g * 1.1, ground, nocc=1, config=EOMConfig(), cc_config=cfg)
    assert not mismatch.ground_valid


@pytest.mark.parametrize("sector", ["ee", "ip", "ea"])
def test_full_doubles_are_validated_before_charged_actions(sector):
    from gradscf import cc

    h = jnp.diag(jnp.array([-1.2, -0.7, 0.3, 0.8]))
    g = jnp.zeros((4, 4, 4, 4))
    ground = cc.run_cc(h, g, nocc=2)
    for corruption in (0.1, jnp.nan):
        bad = ground._replace(t2=ground.t2.at[0, 0, 0, 1].set(corruption))
        result = cc.run_eom(h, g, bad, nocc=2, config=cc.EOMConfig(sector=sector))
        assert not result.ground_valid
        assert not np.any(result.response_valid | result.converged)


def test_amplitude_spaces_and_eom_cache_controls(reference):
    from gradscf import cc
    from gradscf.cc.eom import EOMAmplitudeSpace

    for args in [(1, 1, "unknown"), (-1, 2, "ee"), (True, 2, "ip")]:
        with pytest.raises(ValueError):
            EOMAmplitudeSpace(*args)
    space = EOMAmplitudeSpace(1, 1, "ea")
    with pytest.raises(ValueError):
        space.unpack(jnp.zeros(3))
    with pytest.raises(NotImplementedError):
        space.unpack(jnp.zeros(2, dtype=complex))
    h, g, _, _, _ = reference
    obj = cc.CCSD(cc.CCReference(h, g, 1)).run()
    eom = obj.EOMIP().run()
    obj.residual_tol *= 0.1
    with pytest.raises(RuntimeError, match="changed"):
        eom.check_source()


@pytest.mark.parametrize("sector", ["ee", "ip", "ea"])
def test_iterative_eom_and_full_cc_response(reference, sector):
    from gradscf import cc

    h, g, ground, cfg, _ = reference
    options = cc.EOMConfig(
        sector=sector,
        solver="davidson",
        max_dense=1,
        max_space=16,
        max_cycle=100,
        conv_tol=1e-10,
    )
    result = cc.run_eom(h, g, ground, nocc=1, cc_config=cfg, config=options)
    expected = cc.run_eom(
        h, g, ground, nocc=1, cc_config=cfg, config=cc.EOMConfig(sector=sector)
    )
    assert result.spectrum_complete and np.all(result.response_valid)
    np.testing.assert_allclose(result.energies, expected.energies, atol=1e-11)
    direction = jnp.array([[0.2, 0.04], [0.04, -0.1]])

    def energy(t, options):
        ht = h + t * direction
        gt = g * (1 + 0.02 * t)
        state = cc.run_cc(ht, gt, nocc=1, config=cfg)
        return cc.run_eom(
            ht, gt, state, nocc=1, cc_config=cfg, config=options
        ).energies[0]

    iterative = jax.jit(jax.grad(lambda t: energy(t, options)))(0.0)
    dense = jax.grad(lambda t: energy(t, cc.EOMConfig(sector=sector)))(0.0)
    np.testing.assert_allclose(iterative, dense, atol=1e-9)
    np.testing.assert_allclose(
        iterative, (energy(1e-4, options) - energy(-1e-4, options)) / 2e-4, atol=2e-7
    )


@pytest.fixture(scope="module")
def water_reference():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from gradscf import cc

    mol = pyscf.gto.M(
        atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g", verbose=0
    )
    mf = mol.RHF().run(conv_tol=1e-13)
    h = jnp.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff)
    g = jnp.asarray(ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), h.shape[0]))
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    return h, g, cfg


@pytest.mark.parametrize("sector", ["ee", "ip", "ea"])
def test_incomplete_eom_ritz_response_matches_dense(water_reference, sector):
    from gradscf import cc

    h, g, cfg = water_reference
    options = cc.EOMConfig(
        sector=sector, solver="davidson", max_space=16, max_cycle=150, conv_tol=1e-10
    )
    direction = jnp.diag(jnp.linspace(-0.15, 0.2, h.shape[0]))

    def result(t, options):
        ht, gt = h + t * direction, g * (1 + 0.015 * t)
        ground = cc.run_cc(ht, gt, nocc=5, config=cfg)
        return cc.run_eom(ht, gt, ground, nocc=5, cc_config=cfg, config=options)

    out = jax.jit(lambda t: result(t, options))(0.0)
    assert np.all(out.response_valid)
    assert not out.spectrum_complete
    assert out.iterations > 1
    f = lambda t: result(t, options).energies[0]
    dense = lambda t: result(t, cc.EOMConfig(sector=sector)).energies[0]
    ad = jax.jit(jax.grad(f))(0.0)
    np.testing.assert_allclose(ad, jax.jit(jax.grad(dense))(0.0), atol=2e-7)
    np.testing.assert_allclose(ad, (f(1e-4) - f(-1e-4)) / 2e-4, atol=3e-6)


def test_native_co_degenerate_ee_has_full_rank_dual():
    from gradscf import gto, dft, cc

    mf = dft.RKS(
        gto.M(atom="C 0 0 0; O 0 0 1.128", basis="sto-3g"), xc="hf", conv_tol=1e-12
    ).run()
    ground = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-11, max_cycle=200).run()
    out = (
        cc.EOMEE(
            ground, nroots=3, solver="davidson", max_space=48, max_cycle=180, seed=0
        )
        .run()
        .result
    )
    assert np.all(out.converged)
    assert not np.any(out.response_valid)
    np.testing.assert_allclose(
        out.energies,
        [0.3322036425819134, 0.3322036425819358, 0.411352356441813],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        out.left_vectors.T @ out.right_vectors, np.eye(3), atol=1e-10, rtol=0
    )
    assert np.linalg.svd(out.left_vectors, compute_uv=False)[-1] > 0.5
