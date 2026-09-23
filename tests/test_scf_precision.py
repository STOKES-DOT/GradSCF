"""Shared stationarity, stability and explicit multistart precision controls."""

import numpy as np
import pytest
from gradscf import gto, dft


def test_restricted_internal_stability_and_packed_integrals():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g")
    mf = dft.RKS(mol, xc="hf", conv_tol=1e-12).run()
    check = mf.stability(solver="dense")
    assert check.stationary and check.stable is False
    assert check.minimum_curvature < -1e-3
    assert check.spectrum_certified
    rejected = mf.multistart(amplitudes=(), require_stable=True)
    assert rejected.selected is None and not rejected.converged
    assert rejected.attempts[0].converged and rejected.attempts[0].stable is False
    chosen = mf.multistart(amplitudes=(0.15, 0.4), seed=20260923).selected
    stable = chosen.stability(solver="dense")
    assert stable.stable and stable.stationary
    assert np.max(stable.residual_norms) < 1e-7
    # Orbital response must not require expanding the compressed native ERIs.
    assert mf._scf_inputs.eri is None
    assert mf._scf_inputs.response_eri_pair_matrix() is not None
    diagnostic = chosen.diagnostics()
    assert diagnostic.stationary
    assert diagnostic.gradient_norm < chosen.conv_tol_grad
    assert diagnostic.orthogonality_error < 1e-10
    assert diagnostic.electron_count_error < 1e-10
    chosen.conv_tol *= 0.1
    with pytest.raises(RuntimeError, match="changed"):
        chosen.diagnostics()


def test_multiseed_baseline_once_and_metadata():
    mf = dft.RKS(
        gto.M(atom="F 0 0 0; F 0 0 2.2", basis="sto-3g"),
        xc="hf",
        conv_tol=1e-12,
        conv_tol_density=1e-10,
        conv_tol_grad=1e-9,
        max_cycle=250,
    )
    result = mf.multistart(amplitudes=(0.5, 1.0), seeds=(0, 1), require_stable=True)
    assert len(result.attempts) == 5
    assert result.attempts[0].seed is None
    assert [a.seed for a in result.attempts[1:]] == [0, 0, 1, 1]
    assert result.seeds == (0, 1)
    assert result.converged
    assert result.attempts[result.selected_index].stable is True
    np.testing.assert_allclose(result.selected.e_tot, -195.67162624073558, atol=1e-9)
    assert mf.e_tot is None and mf.scf_result is None
    for seeds in ((), (0, 0), (-1,), (True,)):
        with pytest.raises(ValueError):
            mf.multistart(seeds=seeds)


def test_precision_layer_report_detects_tight_scf_target():
    from gradscf import cc

    mf = dft.RKS(
        gto.M(atom="He 0 0 0; He 0 0 3", basis="6-31g"), xc="hf", conv_tol=1e-12
    ).run()
    obj = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-11).run()
    eom = obj.EOMIP(nroots=3).run()
    diagnostic = eom.diagnostics(scf_gradient_tol=1e-11)
    assert diagnostic.cc is obj.result and diagnostic.eom is eom.result
    assert diagnostic.cc_ok and diagnostic.eom_ok
    assert not diagnostic.scf.stationary
    assert diagnostic.all_passed is False
    assert diagnostic.scf.gradient_norm > 1e-11
    # No fake SCF estimate when the functional input is only an MO Hamiltonian.
    explicit = (
        cc.CCSD(obj.reference, conv_tol=1e-12, residual_tol=1e-11).run().EOMIP().run()
    )
    assert explicit.diagnostics().scf is None
    assert explicit.diagnostics().all_passed is None
    with pytest.raises(ValueError):
        explicit.diagnostics(scf_gradient_tol=-1.0)
    obj.t1 = obj.t1 + 0.01
    with pytest.raises(RuntimeError, match="amplitudes changed"):
        eom.diagnostics()


def test_restricted_curvatures_match_independent_pyscf_hessian():
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto, scf as pscf
    from pyscf.soscf import newton_ah

    atom = "O 0 0 0; H 0 -.757 .587; H 0 .757 .587"
    mf = dft.RKS(gto.M(atom=atom, basis="sto-3g"), xc="hf", conv_tol=1e-12).run()
    result = mf.stability(solver="dense")
    reference = pscf.RHF(pgto.M(atom=atom, basis="sto-3g", verbose=0))
    reference.mo_coeff = np.asarray(mf.mo_coeff)
    reference.mo_occ = np.asarray(mf.mo_occ)
    _, hop, diagonal = newton_ah.gen_g_hop_rhf(
        reference, reference.mo_coeff, reference.mo_occ
    )
    hessian = np.column_stack([2 * hop(v) for v in np.eye(len(diagonal))])
    np.testing.assert_allclose(
        result.eigenvalues, np.linalg.eigvalsh(hessian)[:3], atol=2e-9, rtol=0
    )
    failed_dense = mf.stability(solver="dense", eigensolver_tol=1e-30)
    assert failed_dense.stable is None
    assert (
        not failed_dense.eigensolver_converged and not failed_dense.spectrum_certified
    )
    iterative = mf.stability()
    assert not iterative.spectrum_certified
    np.testing.assert_allclose(
        iterative.eigenvalues, result.eigenvalues, atol=2e-8, rtol=0
    )
    unresolved = mf.stability(max_cycle=1)
    assert unresolved.stable is None
    mf.mo_coeff = mf.mo_coeff.at[0, 0].add(0.01)
    with pytest.raises(RuntimeError, match="orbital arrays changed"):
        mf.stability()


def test_nonstationary_curvature_is_not_called_stable():
    from dataclasses import replace
    from gradscf.scf import run_uhf_from_integrals, uhf_stability
    import jax.numpy as jnp

    result = run_uhf_from_integrals(
        overlap=jnp.eye(2),
        hcore=jnp.diag(jnp.array([-1.0, 1.0])),
        eri=jnp.zeros((2, 2, 2, 2)),
        nalpha=1,
        nbeta=1,
        nuclear_repulsion=0.0,
    )
    angle = 0.1
    c = jnp.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    perturbed = replace(result, mo_coeff_alpha=c, mo_coeff_beta=c)
    check = uhf_stability(perturbed, eri=jnp.zeros((2, 2, 2, 2)))
    assert check.stable is None and not check.stationary
    assert check.gradient_norm > 0.1
