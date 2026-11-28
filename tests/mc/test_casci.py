"""Public CASCI with fixed-spin CSFs and folded molecular integrals."""

import numpy as np
import pytest

from gradscf import gto, scf


@pytest.mark.parametrize(
    "geometry,basis,spin2,nelecas,expected",
    [
        ("H 0 0 0; H 0 0 .74", "sto-3g", 0, (1, 1), -1.137283834488502),
        ("H 0 0 0; H 0 0 .74", "sto-3g", 2, (2, 0), -.530773357001458),
        ("H 0 0 0; H 0 0 1.4", "6-31g", 0, (1, 1), -1.051606680216211),
        ("H 0 0 0; H 0 0 .90; H .85 0 .31", "6-31g", 1, (2, 1), -1.510677772022706),
    ],
)
def test_casci_fixed_spin_molecular_references(geometry, basis, spin2, nelecas, expected):
    from gradscf import mc

    mol = gto.M(atom=geometry, basis=basis, spin=spin2, cart=False)
    hf = scf.ROHF if spin2 else scf.RHF
    mf = hf(mol, conv_tol=1e-13, conv_tol_grad=1e-9).run()
    assert mf.converged
    mc_obj = mc.CASCI(mf, ncas=sum(nelecas), nelecas=nelecas, spin2=spin2, solver="dense")
    assert mc_obj.run() is mc_obj
    assert mc_obj.converged
    np.testing.assert_allclose(mc_obj.e_tot, expected, atol=2e-9, rtol=0)
    np.testing.assert_allclose(mc_obj.spin_square()[0], spin2 * (spin2 + 2) / 4, atol=1e-9)
    d1, d2 = mc_obj.make_rdm12()
    da, db = mc_obj.make_rdm1s()
    np.testing.assert_allclose(d1, da + db, atol=1e-10)
    np.testing.assert_allclose(np.trace(d1), sum(nelecas), atol=1e-10)
    np.testing.assert_allclose(np.einsum("pprr->", d2), sum(nelecas) * (sum(nelecas) - 1), atol=1e-10)


def test_casci_active_validation_and_covariant_source():
    from gradscf import mc

    mf = scf.RHF(gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g"))
    with pytest.raises(RuntimeError, match="converge"):
        mc.CASCI(mf, 2, (1, 1), spin2=0).run()
    mf.run()
    original_coeff = mf.mo_coeff
    original_result = mf.scf_result
    for active in ((0, 2), (1, 1), (0, 99), (1,)):
        with pytest.raises(ValueError, match="active"):
            mc.CASCI(mf, 2, (1, 1), spin2=0, active=active).run()
    obj = mc.CASCI(mf, 2, (1, 1), spin2=0, solver="dense").run()
    assert obj.result.core == (0,)
    assert obj.result.active == (1, 2)
    assert obj.result.mo_coeff is not None
    np.testing.assert_allclose(obj.e_tot, -7.862128833372132, atol=2e-9, rtol=0)
    assert mf.mo_coeff is original_coeff and mf.scf_result is original_result


def test_casci_invalid_integrals_and_degenerate_derivative():
    import jax
    import jax.numpy as jnp
    from gradscf import mc
    from gradscf.solvers import EigenSolverConfig

    space = mc.make_csf_space(2, (1, 1), 0)
    cfg = EigenSolverConfig(method="dense")
    h = jnp.zeros((2, 2))
    eri = jnp.zeros((2,) * 4)
    assert not mc.solve_casci(h.at[0, 1].set(.1), eri, space, config=cfg).converged
    solve = lambda nuclear: mc.solve_casci(h, eri, space, ecore=nuclear, config=cfg).total_energy
    assert float(solve(0.)) == 0.
    assert np.isnan(jax.grad(solve)(0.))


@pytest.mark.parametrize(
    "geometry,basis,spin,nelecas,expected",
    [
        ("Li 0 0 0; H 0 0 1.6", "sto-3g", 0, (1, 1), -7.881045251314651),
        ("H 0 0 0; H 0 0 .90; H .85 0 .31", "6-31g", 1, (2, 1), -1.524105411664464),
    ],
)
def test_casscf_facade_preserves_hf_source(geometry, basis, spin, nelecas, expected):
    from gradscf import mc

    mol = gto.M(atom=geometry, basis=basis, spin=spin, cart=False)
    hf = scf.ROHF if spin else scf.RHF
    mf = hf(mol, conv_tol=1e-13, conv_tol_grad=1e-9).run()
    original = (mf.mo_coeff, mf.mo_occ, mf.e_tot, mf.scf_result)
    answer = mc.CASSCF(mf, sum(nelecas), nelecas, spin2=spin, solver="dense").run()
    assert answer.converged
    np.testing.assert_allclose(answer.e_tot, expected, atol=2e-8, rtol=0)
    assert all(a is b for a, b in zip(original, (mf.mo_coeff, mf.mo_occ, mf.e_tot, mf.scf_result)))
