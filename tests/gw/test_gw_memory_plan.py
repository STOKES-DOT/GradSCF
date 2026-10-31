"""Shared GW budgets change tiles, never the requested physical spaces."""
from importlib import import_module, util
import numpy as np
import pytest


def _planner():
    assert util.find_spec("gradscf.gw._memory") is not None, "GW needs a shared working-memory plan"
    return import_module("gradscf.gw._memory").plan_gw


def test_gw_plan_accounts_for_complex_workspace_and_selected_residents():
    plan = _planner()
    factors = np.zeros((128, 32, 32))
    coeff = np.eye(32)
    controls = dict(nocc=8, n_targets=4, n_g=32, screening_shape=(8, 24), nw=40)
    large = plan(factors, coeff, **controls, max_memory=100)
    small = plan(factors, coeff, **controls, max_memory=10)
    assert large.resident_bytes >= factors.nbytes+coeff.nbytes
    assert 1 <= small.frequency_batch <= large.frequency_batch <= 40
    assert 1 <= small.rhs_batch <= large.rhs_batch <= 128
    assert large.working_bytes+large.resident_bytes <= 80_000_000
    assert small.working_bytes+small.resident_bytes <= 8_000_000


def test_gw_plan_rejects_impossible_minimum_before_allocating():
    with pytest.raises(ValueError, match="budget|memory"):
        _planner()(np.zeros((128, 32, 32)), np.eye(32), nocc=8, n_targets=4,
                   n_g=32, screening_shape=(8, 24), nw=40, max_memory=.01)


def test_gpu_dense_residue_is_retained_only_when_the_whole_batch_fits(monkeypatch):
    from types import SimpleNamespace
    module = import_module("gradscf.gw._memory")
    monkeypatch.setattr(module, "array_device", lambda *args: SimpleNamespace(platform="gpu"))
    args = (np.zeros((128, 32, 32)), np.eye(32))
    controls = dict(nocc=8, n_targets=4, n_g=32, screening_shape=(8, 24), nw=40)
    assert not module.plan_gw(*args, **controls, max_memory=1000).projected_residue
    assert module.plan_gw(*args, **controls, max_memory=10).projected_residue


@pytest.mark.parametrize("resolvent", [False, True])
def test_array_driver_checks_budget_for_selected_and_full_routes(resolvent):
    from gradscf.gw import g0w0_cd_restricted
    with pytest.raises(ValueError, match="budget|memory"):
        g0w0_cd_restricted(mo_energy=np.array([-.8, .3, .9]), mo_coeff=np.eye(3),
            nocc=1, df_factors=np.ones((2, 3, 3))*.01, fock_matrix=np.eye(3),
            hcore_matrix=np.zeros((3, 3)), density_matrix=np.diag([2., 0., 0.]),
            nw=8, max_memory=1e-6, resolvent_expansion=resolvent)


def test_gmres_charges_the_promoted_real_dtype():
    from gradscf.solvers import LinearSolverConfig
    plan = _planner()
    config = LinearSolverConfig(method="gmres", restart=20)
    settings = dict(nocc=8, n_targets=4, n_g=32, screening_shape=(8, 24), nw=40,
                    max_memory=10., screening_config=config)
    mixed = plan(np.zeros((128, 32, 32), dtype=np.float32), np.eye(32), **settings)
    double = plan(np.zeros((128, 32, 32), dtype=np.float64), np.eye(32), **settings)
    assert mixed.working_bytes == double.working_bytes
    assert mixed.resident_bytes > double.resident_bytes  # original + promoted input can coexist


def test_dense_frequency_batch_charges_weighted_transition_factors():
    q, n, no, nv = 32, 256, 128, 128
    plan = _planner()(np.zeros((q, n, n)), np.eye(n), nocc=no,
        n_targets=1, n_g=n, screening_shape=(no, nv), nw=100, max_memory=100.)
    weighted = 2*q*no*nv*16
    assert plan.working_bytes >= plan.frequency_batch*weighted
    assert plan.resident_bytes+plan.working_bytes <= 80_000_000
