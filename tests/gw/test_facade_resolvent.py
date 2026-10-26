"""Exact transition-space RPA is used only for bounded CPU facade layouts.

Float64, Hartree. Direct and resolvent paths share the original six-seed QP
equations, windows and finite-eta convention; no screening modes are fitted.
"""
import importlib
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def small_rhf():
    from gradscf import gto, scf

    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g"), conv_tol=1e-12).run()
    assert mf.converged
    return mf


def _assert_qp_equal(actual, expected, *, atol=3e-9):
    for name in ("mo_energy", "mo_coeff", "sigma_qp", "qp_residual", "qp_weight",
                 "qp_on_shell", "qp_linearized", "screening_energy"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   atol=atol, rtol=3e-9, equal_nan=True)
    for name in ("converged", "converged_mask", "qp_computed_mask", "qp_candidate_count", "qp_branch_valid"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("g_orbitals", "screening_occupied", "screening_virtual", "method", "nw"):
        assert getattr(actual, name) == getattr(expected, name)


@pytest.mark.parametrize("eta", [.001, .03])
def test_small_cpu_facade_preserves_direct_qp_diagnostics(small_rhf, eta):
    from gradscf.gw import GW, g0w0_cd_restricted

    obj = GW(small_rhf, nw=8, eta=eta, g_orbitals=(1, 0),
             screening_occupied=(0,), screening_virtual=(1,))
    r = small_rhf.scf_result
    direct = g0w0_cd_restricted(mo_energy=r.mo_energy, mo_coeff=r.mo_coeff, nocc=1,
        df_factors=obj._df_factors(), fock_matrix=r.fock_matrix, hcore_matrix=r.hcore_matrix,
        density_matrix=r.density_matrix, nw=8, eta=eta, orbs=(1, 0),
        g_orbitals=(1, 0), screening_occupied=(0,), screening_virtual=(1,),
        resolvent_expansion=False)
    actual = obj.run(orbs=(1, 0)).result
    jax.block_until_ready(actual)
    assert bool(actual.converged)
    _assert_qp_equal(actual, direct)


def test_small_cpu_facade_does_not_trace_auxiliary_residue_solve(small_rhf, monkeypatch):
    from gradscf.gw import GW

    rgw = importlib.import_module("gradscf.gw.rgw")
    self_energy = importlib.import_module("gradscf.gw.self_energy")
    # A previously compiled direct executable would hide Python monkeypatches.
    # Clear this one facade executable cache, not process-wide JAX caches.
    rgw._run_g0w0_numerics.clear_cache()
    def forbidden(*args, **kwargs):
        raise AssertionError("eligible CPU facade traced the auxiliary residue solve")
    monkeypatch.setattr(self_energy, "_response_solve_one", forbidden)
    result = GW(small_rhf, nw=8, eta=.02).run().result
    jax.block_until_ready(result)
    assert bool(result.converged)
    np.testing.assert_array_equal(result.qp_computed_mask, [True, True])
    assert np.max(np.abs(np.asarray(result.sigma_qp))) > 1e-6


def test_zero_eta_inactive_dark_pole_resolvent_matches_direct():
    from gradscf.gw import g0w0_cd_restricted
    from gradscf.integrals.molecular.jk import build_jk_from_df

    # The OV vertex is zero. Wc is exactly zero, including at the removable
    # bare transition pole sampled by an inactive internal-G lane.
    energy = jnp.array([-.5, .5])
    factors = jnp.array([[[.3, 0.], [0., .2]]])
    density = jnp.diag(jnp.array([2., 0.]))
    j, k = build_jk_from_df(factors, density)
    arguments = dict(mo_energy=energy, mo_coeff=jnp.eye(2), nocc=1, df_factors=factors,
        fock_matrix=jnp.diag(energy), hcore_matrix=jnp.diag(energy)-j+k/2,
        density_matrix=density, nw=8, eta=0., evaluate_only=True)
    direct = g0w0_cd_restricted(**arguments, resolvent_expansion=False)
    exact = g0w0_cd_restricted(**arguments, resolvent_expansion=True)
    assert np.isfinite(np.asarray(direct.sigma_qp)).all()
    assert np.isfinite(np.asarray(exact.sigma_qp)).all()
    _assert_qp_equal(exact, direct, atol=2e-12)


@dataclass(frozen=True)
class _LayoutDevice:
    platform: str
    id: int = 0


class _PlacedArray(np.ndarray):
    """Virtual placement for dispatch tests; no GPU or quantum solve is used."""
    def __new__(cls, value, platform):
        array = np.asarray(value).view(cls)
        array._platform = platform
        return array

    def __array_finalize__(self, original):
        self._platform = getattr(original, "_platform", "cpu")

    @property
    def device(self):
        return _LayoutDevice(self._platform)

    def devices(self):
        return {self.device}


def _metadata_reference(nmo, nocc, platform, amplitude=0.):
    """Facade metadata only: the spy replaces every physical numerical solve."""
    cpu = jax.local_devices(backend="cpu")[0]
    placed = (lambda value: jax.device_put(value, cpu)) if platform == "cpu" else (
        lambda value: _PlacedArray(value, platform))
    energy = placed(np.linspace(-1., 1., nmo)+.03*amplitude)
    coefficients = placed(np.eye(nmo))
    occupations = np.r_[np.full(nocc, 2.), np.zeros(nmo-nocc)]
    zero = placed(np.zeros((nmo, nmo)))
    factors = placed(np.full((1, nmo, nmo), amplitude))
    result = SimpleNamespace(mo_energy=energy, mo_coeff=coefficients, mo_occ=occupations,
        fock_matrix=zero, hcore_matrix=zero, density_matrix=placed(np.diag(occupations)))
    key = (nmo, nocc, platform, amplitude)
    return SimpleNamespace(mo_energy=energy, mo_coeff=coefficients, mo_occ=occupations,
        scf_result=result, converged=True, reference=None,
        _scf_inputs=SimpleNamespace(df_factors=factors, eri_pair_matrix=None),
        _scf_signature=lambda: key, _cached_scf_key=key)


def _facade_dispatch(monkeypatch, *, platform, nmo, nocc, occupied=None, virtual=None,
                     amplitude=0., operand_override=None):
    rgw = importlib.import_module("gradscf.gw.rgw")
    from gradscf.gw.types import GWResult

    calls = []
    def spy(**arguments):
        calls.append(arguments)
        return GWResult(mo_energy=arguments["mo_energy"], mo_coeff=arguments["mo_coeff"],
                        converged=True)
    monkeypatch.setattr(rgw, "_run_g0w0_numerics", spy)
    source = _metadata_reference(nmo, nocc, platform, amplitude)
    if operand_override is not None:
        name, placement = operand_override
        holder = source._scf_inputs if name == "df_factors" else source.scf_result
        original = getattr(holder, name)
        value = (np.asarray(original) if placement == "numpy" else
                 _PlacedArray(original, placement))
        setattr(holder, name, value)
        if name in ("mo_energy", "mo_coeff"):
            setattr(source, name, value)
    rgw.GW(source, nw=8, eta=.003, screening_occupied=occupied,
           screening_virtual=virtual).run(orbs=(0,))
    assert len(calls) == 1
    assert calls[0]["qp_solver"] == "secant"
    assert calls[0]["orbs"] == (0,) and calls[0]["eta"] == .003
    return bool(calls[0].get("resolvent_expansion", False))


@pytest.mark.parametrize("platform,nmo,nocc,occupied,virtual,expected", [
    ("cpu", 4, 2, None, None, True),
    ("gpu", 4, 2, None, None, False),
    ("cpu", 34, 17, None, None, False),
    ("cpu", 32, 16, None, None, True),
    ("cpu", 500, 1, (0,), tuple(range(1, 81)), True),
    ("cpu", 500, 1, (0,), tuple(range(1, 82)), False),
    ("cpu", 50, 25, (0,), (25, 26), True),
    ("cpu", 4, 2, (), None, False),
    ("cpu", 4, 2, None, (), False),
], ids=["small-cpu", "gpu", "large-transition", "transition-boundary",
        "coupling-boundary", "large-coupling", "selected-small-screening",
        "empty-occupied", "empty-virtual"])
def test_facade_resolvent_policy_uses_only_static_layout_and_device(
        monkeypatch, platform, nmo, nocc, occupied, virtual, expected):
    assert _facade_dispatch(monkeypatch, platform=platform, nmo=nmo, nocc=nocc,
                            occupied=occupied, virtual=virtual) is expected


def test_facade_resolvent_policy_does_not_inspect_numerical_values(monkeypatch):
    flags = [_facade_dispatch(monkeypatch, platform="cpu", nmo=4, nocc=2,
                              amplitude=value) for value in (0., .17)]
    assert flags == [True, True]


@pytest.mark.parametrize("name", ["mo_coeff", "df_factors", "fock_matrix",
                                 "hcore_matrix", "density_matrix"])
def test_cpu_energy_does_not_enable_resolvent_with_other_gpu_operands(monkeypatch, name):
    assert not _facade_dispatch(monkeypatch, platform="cpu", nmo=4, nocc=2,
                                operand_override=(name, "gpu"))


def test_uncommitted_numpy_energy_keeps_direct_family(monkeypatch):
    assert not _facade_dispatch(monkeypatch, platform="cpu", nmo=4, nocc=2,
                                operand_override=("mo_energy", "numpy"))


def test_global_gpu_default_keeps_direct_even_with_cpu_inputs(monkeypatch):
    rgw = importlib.import_module("gradscf.gw.rgw")
    actual_jax = rgw.jax
    class Config:
        jax_default_device = _LayoutDevice("gpu")
        def __getattr__(self, name):
            return getattr(actual_jax.config, name)
    class Jax:
        config = Config()
        def __getattr__(self, name):
            return getattr(actual_jax, name)
    # Only this module's device-query view changes. No global JAX setting or
    # accelerator backend is initialized, and the numerical driver is a spy.
    monkeypatch.setattr(rgw, "jax", Jax())
    assert not _facade_dispatch(monkeypatch, platform="cpu", nmo=4, nocc=2)


def test_global_cpu_default_allows_bounded_cpu_resolvent(monkeypatch):
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        assert _facade_dispatch(monkeypatch, platform="cpu", nmo=4, nocc=2)


def _joint_model(theta, *, resolvent):
    from gradscf.gw import g0w0_cd_restricted
    from gradscf.integrals.molecular.jk import build_jk_from_df

    energy = jnp.array([-1.2, -.7, .2, .6, .9])
    raw = jnp.asarray(np.random.default_rng(127).normal(size=(3, 5, 5)))*.12
    factors = (raw+raw.swapaxes(1, 2))*.5
    density = jnp.diag(jnp.array([2., 2., 0., 0., 0.]))
    j, k = build_jk_from_df(factors, density)
    fock = jnp.diag(energy)
    hcore = fock-j+k/2
    angle = .03*theta[1]
    rotation = jnp.eye(5).at[0, 0].set(jnp.cos(angle)).at[2, 2].set(jnp.cos(angle))
    rotation = rotation.at[0, 2].set(-jnp.sin(angle)).at[2, 0].set(jnp.sin(angle))
    energy_direction = jnp.array([-.03, .02, .01, -.02, .04])
    pole_direction = jnp.array([.02, -.01, .03, .01, -.02])
    screen_direction = jnp.array([-.01, .02, -.02, .03, .01])
    delta = jnp.diag(jnp.array([.03, -.02, .01, -.015, .025]))
    return g0w0_cd_restricted(mo_energy=energy+theta[0]*energy_direction,
        mo_coeff=rotation, nocc=2, df_factors=factors*(1+.07*theta[1]),
        fock_matrix=fock+theta[0]*delta, hcore_matrix=hcore-.3*theta[0]*delta,
        density_matrix=density+theta[0]*delta, nw=12, eta=.03+.01*theta[3],
        mo_energy_poles=energy+theta[2]*pole_direction,
        screening_energy=energy+theta[2]*screen_direction, orbs=(1, 3),
        g_orbitals=(4, 0, 2), screening_occupied=(1, 0), screening_virtual=(4, 2),
        resolvent_expansion=resolvent, qp_tol=1e-11)


def test_exact_resolvent_joint_physical_leaf_eta_gradient_and_hvp_match_direct():
    selected = jnp.array([1, 3])
    def observable(theta, resolvent):
        result = _joint_model(theta, resolvent=resolvent)
        return (jnp.vdot(result.mo_energy[selected], jnp.array([1., .7]))
                +.01*jnp.sum(jnp.abs(result.sigma_qp[selected])**2))
    direct = jax.jit(lambda theta: observable(theta, False))
    exact = jax.jit(lambda theta: observable(theta, True))
    point = jnp.zeros(4)
    _assert_qp_equal(_joint_model(point, resolvent=True), _joint_model(point, resolvent=False))
    direct_gradient = jax.jit(jax.grad(direct))
    exact_gradient = jax.jit(jax.grad(exact))
    gradient = exact_gradient(point)
    np.testing.assert_allclose(exact(point), direct(point), atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(gradient, direct_gradient(point), atol=3e-10, rtol=3e-6)
    assert np.linalg.norm(np.asarray(gradient)) > 1e-6
    for h in (2e-4, 1e-4):
        fd = jnp.stack([(exact(point+h*d)-exact(point-h*d))/(2*h) for d in jnp.eye(4)])
        np.testing.assert_allclose(gradient, fd, atol=3e-8, rtol=3e-5)
    direction = jnp.array([.2, -.3, .1, .4])
    tangent = jax.jvp(exact, (point,), (direction,))[1]
    np.testing.assert_allclose(tangent, jnp.vdot(gradient, direction), atol=3e-10)
    hvp = jax.jvp(exact_gradient, (point,), (direction,))[1]
    reference_hvp = jax.jvp(direct_gradient, (point,), (direction,))[1]
    np.testing.assert_allclose(hvp, reference_hvp, atol=3e-8, rtol=3e-5)
    h = 1e-4
    np.testing.assert_allclose(hvp,
        (exact_gradient(point+h*direction)-exact_gradient(point-h*direction))/(2*h),
        atol=3e-7, rtol=3e-5)
