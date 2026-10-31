"""Fixed-orbital GW iterations reuse data, never a stale energy spectrum."""
import jax.numpy as jnp
import numpy as np
import pytest


def inputs():
    e = jnp.array([-.6, .7])
    return dict(mo_energy=e, mo_coeff=jnp.eye(2), nocc=1,
        df_factors=jnp.array([[[.16, .08], [.08, .12]], [[.07, -.04], [-.04, .11]]]),
        fock_matrix=jnp.diag(e), hcore_matrix=jnp.diag(e),
        density_matrix=jnp.diag(jnp.array([2., 0.])), nw=12, max_iter=60, tol=1e-11)


@pytest.mark.parametrize("update_w", [False, True])
def test_evgw_prepares_fixed_vertices_once(monkeypatch, update_w):
    from gradscf.gw import evgw_cd_restricted, g0w0_cd_restricted, integrals, screened
    from gradscf.gw.g0w0 import _run_g0w0_numerics
    _run_g0w0_numerics.clear_cache()
    prepared, screened_at = [], []
    original, screening = integrals.prepare_selected_factors, screened.screened_w_selected
    def prepare(*args, **kwargs):
        prepared.append(1)
        return original(*args, **kwargs)
    def evaluate(*args, **kwargs):
        screened_at.append(1)
        return screening(*args, **kwargs)
    monkeypatch.setattr(integrals, "prepare_selected_factors", prepare)
    monkeypatch.setattr(screened, "screened_w_selected", evaluate)
    kw = inputs()
    result = evgw_cd_restricted(**kw, update_w=update_w)
    assert bool(result.converged)
    assert len(prepared) == 1, "Energy updates must reuse fixed ordered MO vertices"
    assert len(screened_at) == 1  # Trace once; the independent residual below checks live W.
    base = {k: v for k, v in kw.items() if k not in ("max_iter", "tol")}
    check = g0w0_cd_restricted(**base, mo_energy_poles=result.mo_energy,
        screening_energy=None if update_w else kw["mo_energy"], evaluate_only=True)
    np.testing.assert_allclose(check.qp_residual, result.qp_residual, atol=1e-12)


def test_evgw_accepts_shared_screening_budget():
    from gradscf.gw import evgw_cd_restricted
    from gradscf.solvers import LinearSolverConfig
    import inspect
    assert "max_memory" in inspect.signature(evgw_cd_restricted).parameters
    with pytest.raises(ValueError, match="budget|memory"):
        evgw_cd_restricted(**inputs(), max_memory=1e-6)
    dense = evgw_cd_restricted(**inputs())
    iterative = evgw_cd_restricted(**inputs(), max_memory=10,
        screening_config=LinearSolverConfig(method="gmres", rtol=1e-12, atol=1e-14, restart=8))
    np.testing.assert_allclose(iterative.mo_energy, dense.mo_energy, atol=1e-10)


def test_evgw_preparation_uses_the_driver_float64_contract():
    from gradscf.gw import evgw_cd_restricted
    kw = inputs()
    single = {k: v.astype(jnp.float32) if hasattr(v, "astype") else v for k, v in kw.items()}
    expected = evgw_cd_restricted(**{k: v.astype(jnp.float64) if hasattr(v, "astype") else v
                                    for k, v in single.items()})
    actual = evgw_cd_restricted(**single)
    assert actual.mo_energy.dtype == jnp.float64
    np.testing.assert_allclose(actual.mo_energy, expected.mo_energy, atol=1e-12)


def test_evgw_does_not_retrace_each_outer_energy_update(monkeypatch, record_property):
    from gradscf.gw import evgw_cd_restricted, g0w0
    compiled = getattr(g0w0, "_run_g0w0_numerics", None)
    if compiled is not None:
        compiled.clear_cache()
    calls, original = [], g0w0._qp_loop
    def counted(**kwargs):
        calls.append(1)
        return original(**kwargs)
    monkeypatch.setattr(g0w0, "_qp_loop", counted)
    result = evgw_cd_restricted(**inputs())
    assert bool(result.converged)
    assert len(calls) == 1, "Outer energy updates must reuse the numerical program"
    record_property("cold_qp_traces", len(calls))
    changed = inputs()
    changed["mo_energy"] = changed["mo_energy"]+jnp.array([.01, -.02])
    updated = evgw_cd_restricted(**changed)
    assert len(calls) == 1, "A new input object must not recompile the same layout"
    record_property("new_object_qp_traces_total", len(calls))
    assert np.max(np.abs(np.asarray(result.mo_energy-updated.mo_energy))) > 1e-5
    compiled.clear_cache()
    fresh = evgw_cd_restricted(**changed)
    np.testing.assert_allclose(updated.mo_energy, fresh.mo_energy, atol=1e-12)
