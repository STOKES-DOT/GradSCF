"""Native UHF public response routing and its exact-exchange forward limit."""

import numpy as np
import jax.numpy as jnp
import pytest


@pytest.fixture(scope="module")
def references():
    pyscf = pytest.importorskip("pyscf")
    from gradscf import gto, scf

    atom = "H 0 0 0; H 0 0 .85; H 0 0 1.9"
    mf = scf.UHF(
        gto.M(atom=atom, basis="sto-3g", spin=1),
        conv_tol=1e-12,
        conv_tol_density=1e-11,
        conv_tol_grad=1e-10,
        max_cycle=200,
    ).run()
    ref = (
        pyscf.gto.M(atom=atom, basis="sto-3g", spin=1, verbose=0)
        .UHF()
        .run(conv_tol=1e-12, conv_tol_grad=1e-9)
    )
    assert mf.converged and ref.converged
    return mf, ref


@pytest.mark.parametrize("method,reference_method", [("TDA", "TDA"), ("TDDFT", "TDHF")])
def test_native_uhf_response_matches_pyscf(references, method, reference_method):
    mf, ref = references
    calc = getattr(mf, method)(nstates=2, davidson_tol=1e-10, davidson_max_iter=200)
    result = calc.kernel()
    expected = (
        getattr(ref, reference_method)()
        .set(nstates=2, conv_tol=1e-9, max_cycle=200)
        .run()
    )
    assert np.all(result.converged) and np.all(expected.converged)
    np.testing.assert_allclose(calc.e, expected.e, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        calc.oscillator_strength(), expected.oscillator_strength(), atol=2e-7, rtol=0
    )


def test_unrestricted_hf_semilocal_hvp_is_zero_without_density_evaluation():
    from gradscf.tddft._unrestricted_semilocal_response import (
        UnrestrictedSemilocalResponseFunctional,
    )

    hf = UnrestrictedSemilocalResponseFunctional("hf")
    assert hf.exact_exchange_fraction == 1.0
    ta, tb = jnp.ones((2, 1, 4)), jnp.full((2, 1, 4), 0.3)
    xa, xb = hf.spin_grid_response_hvp(None, ta, tb)
    np.testing.assert_array_equal(xa, 0.0)
    np.testing.assert_array_equal(xb, 0.0)
