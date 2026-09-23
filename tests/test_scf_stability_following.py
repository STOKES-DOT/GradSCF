"""Spin-channel Hessians and shared, explicit negative-mode stabilization."""

import numpy as np
import pytest
from gradscf import dft, gto


def hydrogen(distance):
    return dft.RKS(
        gto.M(atom=f"H 0 0 0; H 0 0 {distance}", basis="sto-3g"),
        xc="hf",
        conv_tol=1e-12,
        conv_tol_density=1e-10,
        conv_tol_grad=1e-9,
        max_cycle=150,
    ).run()


def test_spin_channel_detects_instability_missed_by_internal_space():
    pytest.importorskip("pyscf")
    from pyscf import scf as pscf, gto as pgto
    from pyscf.scf.stability import _gen_hop_rhf_external

    mf = hydrogen(2.5)
    assert mf.stability(solver="dense").stable
    spin = mf.stability(channel="spin", solver="dense")
    assert spin.stable is False and spin.stationary
    assert spin.channel == "spin" and spin.mo_coeff.shape == (2, 2, 2)
    assert spin.direction.shape == (1,)
    reference = pscf.RHF(pgto.M(atom="H 0 0 0; H 0 0 2.5", basis="sto-3g", verbose=0))
    reference.mo_coeff = np.asarray(mf.mo_coeff)
    reference.mo_occ = np.asarray(mf.mo_occ)
    _, _, hop, diagonal = _gen_hop_rhf_external(reference, with_symmetry=False)
    matrix = np.column_stack([4 * hop(v) for v in np.eye(len(diagonal))])
    np.testing.assert_allclose(
        spin.eigenvalues, np.linalg.eigvalsh(matrix)[:3], atol=2e-9, rtol=0
    )
    assert hydrogen(0.74).stability(channel="spin", solver="dense").stable


def test_spin_escape_is_explicit_and_matches_uhf_reference():
    pytest.importorskip("pyscf")
    from pyscf import scf as pscf, gto as pgto

    mf = hydrogen(2.5)
    before = float(mf.e_tot)
    old = np.asarray(mf.mo_coeff).copy()
    result = mf.stabilize(channel="spin", max_restarts=3, step_sizes=(0.5, 1.0))
    assert result.stable and result.status == "stable"
    assert isinstance(result.result, dft.UKS)
    assert result.result.e_tot < before - 0.1
    assert mf.mo_coeff.ndim == 2 and float(mf.e_tot) == before
    np.testing.assert_array_equal(mf.mo_coeff, old)
    assert all(b < a for a, b in zip(result.energy_history, result.energy_history[1:]))
    assert any(t.step < 0 for t in result.attempts) and any(
        t.step > 0 for t in result.attempts
    )
    assert sum(t.accepted for t in result.attempts) == len(result.energy_history) - 1
    ref = pscf.UHF(pgto.M(atom="H 0 0 0; H 0 0 2.5", basis="sto-3g", verbose=0))
    dm = np.asarray(result.result.reference.rdm1)
    ref.kernel(dm0=dm)
    np.testing.assert_allclose(result.result.e_tot, ref.e_tot, atol=1e-9, rtol=0)
    assert result.result.stability(solver="dense").stable
    from gradscf import cc

    with pytest.raises(NotImplementedError):
        cc.CCSD(result.result).run().EOMEE().run()


def test_internal_escape_preserves_restricted_model_and_budget():
    mf = dft.RKS(
        gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g"),
        xc="hf",
        conv_tol=1e-12,
        conv_tol_density=1e-10,
        conv_tol_grad=1e-9,
        max_cycle=150,
    ).run()
    stopped = mf.stabilize(max_restarts=0)
    assert stopped.status == "max_restarts" and not stopped.stable
    assert not stopped.attempts
    fixed = mf.stabilize(max_restarts=4)
    assert fixed.stable and isinstance(fixed.result, dft.RKS)
    np.testing.assert_allclose(fixed.result.e_tot, -107.4965005117978, atol=1e-9)
    assert mf.e_tot > fixed.result.e_tot + 0.7
    for kwargs in (
        {"max_restarts": -1},
        {"step_sizes": ()},
        {"step_sizes": (0.0,)},
        {"step_sizes": (float("nan"),)},
        {"channel": "complex"},
    ):
        with pytest.raises(ValueError):
            mf.stabilize(**kwargs)


def test_zero_spin_gradient_does_not_mask_nonstationary_charge_state():
    from dataclasses import replace

    mf = dft.RKS(
        gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g"),
        xc="hf",
        max_cycle=1,
    ).run()
    # Deliberately inaccurate convergence flag: the full gradient must still
    # prevent classification using a vanishing spin-projected gradient alone.
    mf.converged = True
    mf.scf_result = replace(mf.scf_result, converged=True)
    out = mf.stability(channel="spin", solver="dense")
    assert out.stable is None and not out.stationary
    assert out.gradient_norm > mf.conv_tol_grad
    assert out.direction is None


def test_nonlower_trials_preserve_current_state_and_report_failure():
    mf = dft.RKS(
        gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g"), xc="hf", conv_tol=1e-12
    ).run()
    out = mf.stabilize(max_restarts=2, step_sizes=(1e-14,))
    assert out.status == "no_lower_converged_candidate"
    assert not out.stable and out.restarts == 1
    assert len(out.attempts) == 2 and not any(t.accepted for t in out.attempts)
    assert len(out.energy_history) == 1
    np.testing.assert_allclose(out.result.e_tot, mf.e_tot, atol=1e-10)
    stable = hydrogen(0.74).stabilize(channel="spin", max_restarts=1)
    assert stable.stable and isinstance(stable.result, dft.RKS)
    assert not stable.attempts


def test_unrestricted_facade_rejects_stale_orbitals_and_settings():
    out = hydrogen(2.5).stabilize(channel="spin", max_restarts=2, step_sizes=(1.0,))
    source = out.result
    source.mo_coeff = source.mo_coeff.at[0, 0, 0].add(0.01)
    with pytest.raises(RuntimeError, match="orbital arrays changed"):
        source.stability()
    source.mo_coeff = source.reference.mo_coeff
    source.conv_tol *= 0.1
    with pytest.raises(RuntimeError, match="settings changed"):
        source.stability()


def test_solved_source_is_not_resolved_for_zero_restart_budget(monkeypatch):
    mf = hydrogen(0.74)

    def forbidden(self):
        pytest.fail("stabilize must analyze the existing fresh SCF point")

    monkeypatch.setattr(type(mf), "kernel", forbidden)
    result = mf.stabilize(max_restarts=0, step_sizes=iter((0.5, 1.0)))
    assert result.stable and result.restarts == 0 and not result.attempts
    assert result.result is not mf
    np.testing.assert_array_equal(result.result.mo_coeff, mf.mo_coeff)


def test_water_spin_hessian_matches_independent_response_action():
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto, scf as pscf
    from pyscf.scf.stability import _gen_hop_rhf_external

    atom = "O 0 0 0; H 0 -.757 .587; H 0 .757 .587"
    mf = dft.RKS(gto.M(atom=atom, basis="sto-3g"), xc="hf", conv_tol=1e-12).run()
    check = mf.stability(channel="spin", solver="dense")
    reference = pscf.RHF(pgto.M(atom=atom, basis="sto-3g", verbose=0))
    reference.mo_coeff = np.asarray(mf.mo_coeff)
    reference.mo_occ = np.asarray(mf.mo_occ)
    _, _, hop, diagonal = _gen_hop_rhf_external(reference, with_symmetry=False)
    expected = np.linalg.eigvalsh(
        np.column_stack([4 * hop(v) for v in np.eye(len(diagonal))])
    )
    assert check.stable and check.spectrum_certified
    np.testing.assert_allclose(check.eigenvalues, expected[:3], atol=2e-9, rtol=0)


def test_energy_alias_cannot_corrupt_stabilization_acceptance():
    mf = hydrogen(2.5)
    mf.e_tot -= 10.0
    with pytest.raises(RuntimeError, match="energy changed"):
        mf.stabilize(channel="spin", max_restarts=1, step_sizes=(1.0,))
    mf.e_tot = mf.scf_result.total_energy
    unrestricted = mf.stabilize(
        channel="spin", max_restarts=2, step_sizes=(1.0,)
    ).result
    unrestricted.e_tot += 1.0
    with pytest.raises(RuntimeError, match="energy changed"):
        unrestricted.stability()
    with pytest.raises(RuntimeError, match="energy changed"):
        unrestricted.stabilize()
