"""Physical limits of static TDA-BSE, independent of GW approximation errors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("singlet", [True, False])
def test_bare_screening_hf_limit_matches_cis(singlet):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from gradscf import bse

    mf = (
        pyscf.gto.M(
            atom="H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1", basis="sto-3g", verbose=0
        )
        .RHF()
        .run(conv_tol=1e-13)
    )
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff), 4)
    values, vectors = np.linalg.eigh(eri.reshape(16, 16))
    keep = values > 1e-12
    factors = (vectors[:, keep] * np.sqrt(values[keep])).T.reshape(-1, 4, 4)
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    optical = bse.make_bse_space(4, 2)
    unscreened = bse.make_bse_space(4, 2, occupied=(), virtual=())
    result = bse.run_bse(
        mf.mo_energy,
        mf.mo_energy,
        jnp.asarray(factors),
        optical,
        screening_space=unscreened,
        config=bse.BSEConfig(nroots=3, solver="dense", singlet=singlet),
    )
    td = mf.TDA().set(nstates=3, singlet=singlet, conv_tol=1e-11).run()
    np.testing.assert_allclose(result.excitation_energies, td.e, atol=2e-10, rtol=0)
    dipole = np.einsum(
        "xmn,mi,nj->xij", mf.mol.intor("int1e_r", comp=3), mf.mo_coeff, mf.mo_coeff
    )
    expected = td.oscillator_strength() if singlet else np.zeros(3)
    np.testing.assert_allclose(
        bse.oscillator_strengths(result, dipole, optical), expected, atol=2e-10, rtol=0
    )


def test_independent_transition_and_instability_diagnostics():
    from gradscf import bse

    space = bse.make_bse_space(2, 1)
    e = jnp.array([-1.0, 1.0])
    cfg = bse.BSEConfig(nroots=1, solver="dense")
    independent = bse.run_bse(e, e, jnp.zeros((0, 2, 2)), space, config=cfg)
    assert independent.converged[0] and independent.response_valid[0]
    np.testing.assert_allclose(independent.excitation_energies, [2.0], atol=1e-14)
    factors = jnp.array([[[2.0, 0.0], [0.0, 2.0]]])
    unstable = bse.run_bse(e, e, factors, space, config=cfg)
    assert (
        unstable.converged[0]
        and not unstable.stable[0]
        and not unstable.response_valid[0]
    )
    np.testing.assert_allclose(unstable.excitation_energies, [-2.0], atol=1e-14)
    assert np.isnan(
        bse.oscillator_strengths(unstable, jnp.ones((3, 2, 2)), space)
    ).all()
    assert not np.isfinite(
        jax.grad(
            lambda x: bse.run_bse(
                e * x, e, factors, space, config=cfg
            ).excitation_energies[0]
        )(1.0)
    )


def test_underconverged_roots_do_not_have_valid_derivatives():
    from gradscf import bse

    rng = np.random.default_rng(88)
    l = jnp.asarray(rng.normal(size=(5, 5, 5)) * 0.08)
    l = (l + l.transpose(0, 2, 1)) / 2
    e = jnp.array([-1.1, -0.6, 0.3, 0.8, 1.2])
    space = bse.make_bse_space(5, 2)
    cfg = bse.BSEConfig(nroots=1, max_cycle=1, max_space=3, conv_tol=1e-13)
    result = bse.run_bse(e, e, l, space, config=cfg)
    assert not result.converged[0]
    assert not np.isfinite(
        jax.grad(
            lambda x: bse.run_bse(e, e, l * x, space, config=cfg).excitation_energies[0]
        )(1.0)
    )
