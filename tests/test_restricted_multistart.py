"""Explicit native restricted SCF branch selection, separate from SCF AD."""

import numpy as np
import pytest
from gradscf import gto, dft


def nitrogen(**kwargs):
    return dft.RKS(
        gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g"),
        xc="hf",
        conv_tol=1e-12,
        conv_tol_density=1e-10,
        conv_tol_grad=1e-9,
        **kwargs,
    )


def test_native_multistart_selects_lower_branch_without_mutating_input():
    mf = nitrogen(max_cycle=150).run()
    old_energy = float(mf.e_tot)
    old_coeff = np.asarray(mf.mo_coeff).copy()
    result = mf.multistart(amplitudes=(0.05, 0.15, 0.4), seed=20260923)
    assert result.converged
    assert len(result.attempts) == 4
    assert result.selected is not mf
    assert result.selected_index in (2, 3)
    np.testing.assert_allclose(result.selected.e_tot, -107.4965005117978, atol=1e-9)
    assert old_energy - float(result.selected.e_tot) > 0.7
    assert float(mf.e_tot) == old_energy
    np.testing.assert_array_equal(mf.mo_coeff, old_coeff)
    assert [a.amplitude for a in result.attempts] == [0.0, 0.05, 0.15, 0.4]
    assert all(a.converged for a in result.attempts)
    # An explicitly empty restart list keeps the baseline without claiming a
    # global minimum or silently adding more attempts.
    baseline = mf.multistart(amplitudes=())
    assert len(baseline.attempts) == 1 and baseline.selected_index == 0
    np.testing.assert_allclose(baseline.selected.e_tot, old_energy, atol=1e-10)


def test_unconverged_candidates_are_reported_but_not_selected():
    result = nitrogen(max_cycle=1).multistart(amplitudes=(0.15,), seed=20260923)
    assert len(result.attempts) == 2
    assert not result.converged
    assert result.selected is None and result.selected_index is None
    assert not any(a.converged for a in result.attempts)


def test_multistart_input_controls():
    from gradscf.scf import run_restricted_multistart

    mf = nitrogen()
    for args in (
        {"amplitudes": (-0.1,)},
        {"amplitudes": (float("nan"),)},
        {"amplitudes": (0.0,)},
        {"seed": -1},
        {"seed": True},
    ):
        with pytest.raises(ValueError):
            mf.multistart(**args)
    with pytest.raises(TypeError):
        run_restricted_multistart(object())
