"""Public GW/BSE reuse compiled numerics while physical array inputs stay live.

CPU float64; energies/broadenings are Hartree. Tiny cases test compilation
boundaries and parity, not frequency-window or molecular-basis convergence.
"""
from contextlib import contextmanager
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@contextmanager
def _backend_compilations(monkeypatch):
    from jax._src import compiler

    original = compiler.backend_compile_and_load
    calls = []
    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(compiler, "backend_compile_and_load", counted)
        yield calls


def _assert_qp_equal(actual, expected):
    for name in ("mo_energy", "sigma_qp", "qp_residual", "qp_weight", "qp_on_shell",
                 "qp_linearized", "screening_energy"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   atol=3e-10, rtol=3e-10, equal_nan=True)
    for name in ("converged", "converged_mask", "qp_computed_mask", "qp_candidate_count", "qp_branch_valid"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("g_orbitals", "screening_occupied", "screening_virtual", "method", "nw"):
        assert getattr(actual, name) == getattr(expected, name)


def test_fresh_g0w0_facades_reuse_compilation_and_preserve_qp_diagnostics(monkeypatch):
    from gradscf import gto, scf
    from gradscf.gw import GW, g0w0_cd_restricted

    mean_fields = [scf.RHF(gto.M(atom=f"H 0 0 0; H 0 0 {distance}", basis="sto-3g"),
                           conv_tol=1e-12).run() for distance in (.74, .79)]
    assert all(mf.converged for mf in mean_fields)
    controls = dict(nw=8, g_orbitals=(1, 0), screening_occupied=(0,), screening_virtual=(1,))
    reference_object = GW(mean_fields[1], eta=.002, **controls)
    reference = mean_fields[1].scf_result
    expected = g0w0_cd_restricted(
        mo_energy=reference.mo_energy, mo_coeff=reference.mo_coeff, nocc=1,
        df_factors=reference_object._df_factors(), fock_matrix=reference.fock_matrix,
        hcore_matrix=reference.hcore_matrix, density_matrix=reference.density_matrix,
        eta=.002, orbs=(1, 0), **controls)
    first = GW(mean_fields[0], eta=.001, **controls).run(orbs=(1, 0))
    jax.block_until_ready(first.result)
    with _backend_compilations(monkeypatch) as compilations:
        second = GW(mean_fields[1], eta=.002, **controls).run(orbs=(1, 0))
        jax.block_until_ready(second.result)
    _assert_qp_equal(second.result, expected)
    assert abs(float(second.mo_energy[0]-first.mo_energy[0])) > 1e-5
    assert np.all(second.result.qp_candidate_count > 0)
    assert not compilations, f"Fresh G0W0 inputs triggered {len(compilations)} backend compilations"


def _bse_inputs(t=0.):
    from gradscf.bse import BSEReference

    energy = jnp.array([-.8, .35, .7])
    factors = jnp.array([[[.2, .08, .03], [.08, .13, .04], [.03, .04, .15]],
                         [[.1, .03, -.04], [.03, .09, .02], [-.04, .02, .12]]])
    dipole = jnp.array([[[.1, .2, .12], [.2, .04, .08], [.12, .08, .06]],
                        [[.03, -.08, .11], [-.08, .02, .04], [.11, .04, .09]],
                        [[.02, .05, -.1], [.05, .04, .07], [-.1, .07, .03]]])
    return BSEReference(energy+t*jnp.array([-.03, .02, .04]),
        energy+jnp.array([-.1, -.02, .03])+t*jnp.array([.02, -.01, .03]),
        factors*(1+.1*t), 1, dipole*(1+.03*t),
        jnp.ones(3, dtype=bool), jnp.ones(3, dtype=bool))


def _assert_bse_equal(actual, expected):
    for name in ("excitation_energies", "residual_norms", "min_screening_gap",
                 "stability_margins", "stability_residual_norms"):
        left, right = getattr(actual, name), getattr(expected, name)
        if left is None or right is None:
            assert left is right
        else:
            np.testing.assert_allclose(left, right, atol=3e-11, rtol=3e-10, equal_nan=True)
    # Eigenvector signs are a gauge; compare their projectors/amplitude norms.
    for left, right in zip(jax.tree.leaves((actual.x_amplitudes, actual.y_amplitudes)),
                           jax.tree.leaves((expected.x_amplitudes, expected.y_amplitudes))):
        a, b = np.asarray(left).reshape(left.shape[0], -1), np.asarray(right).reshape(right.shape[0], -1)
        np.testing.assert_allclose(a[:, :, None]*a[:, None, :], b[:, :, None]*b[:, None, :],
                                   atol=3e-11, rtol=3e-10)
    for name in ("converged", "stable", "response_valid", "screening_valid", "stability_certified"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("tda,solver", [(True, "davidson"), (True, "dense"),
                                        (False, "davidson"), (False, "dense")])
def test_fresh_bse_facades_reuse_compilation_and_keep_inputs_dynamic(monkeypatch, tda, solver):
    from gradscf import bse

    first_inputs, second_inputs = _bse_inputs(), _bse_inputs(.07)
    cfg = bse.BSEConfig(nroots=1, tda=tda, solver=solver, conv_tol=1e-11)
    space = bse.make_bse_space(3, 1)
    expected = bse.run_bse(second_inputs.qp_energy, second_inputs.screening_energy,
        second_inputs.mo_factors, space, config=cfg,
        qp_computed_mask=second_inputs.qp_computed_mask,
        qp_converged_mask=second_inputs.qp_converged_mask)
    first = bse.BSE(first_inputs, nroots=1, tda=tda, solver=solver, conv_tol=1e-11).run()
    jax.block_until_ready(first.result)
    with _backend_compilations(monkeypatch) as compilations:
        second = bse.BSE(second_inputs, nroots=1, tda=tda, solver=solver, conv_tol=1e-11).run()
        jax.block_until_ready(second.result)
    _assert_bse_equal(second.result, expected)
    np.testing.assert_allclose(second.oscillator_strength(),
        bse.oscillator_strengths(expected, second_inputs.dipole_mo, space), atol=3e-11, rtol=3e-10)
    assert abs(float(second.e[0]-first.e[0])) > 1e-5
    assert not compilations, f"Fresh BSE inputs triggered {len(compilations)} backend compilations"
    with pytest.raises(ValueError, match="computed, converged"):
        bse.BSE(replace(second_inputs, qp_computed_mask=jnp.array([True, False, True])),
                nroots=1, tda=tda, solver=solver).run()


def _gw_array_inputs():
    from gradscf.integrals.molecular.jk import build_jk_from_df

    energy = jnp.array([-.8, .5])
    factors = jnp.array([[[.45, .12], [.12, .3]], [[.2, -.07], [-.07, .32]]])
    density = jnp.diag(jnp.array([2., 0.]))
    j, k = build_jk_from_df(factors, density)
    return dict(mo_energy=energy, mo_coeff=jnp.eye(2), df_factors=factors,
        fock_matrix=jnp.diag(energy), hcore_matrix=jnp.diag(energy)-j+k/2,
        density_matrix=density, eta=.025)


@pytest.mark.parametrize("leaf", ["mo_energy", "mo_coeff", "df_factors", "fock_matrix",
                                  "hcore_matrix", "density_matrix", "eta"])
def test_g0w0_array_driver_preserves_physical_leaf_response(leaf):
    from gradscf.gw import g0w0_cd_restricted

    base = _gw_array_inputs()
    directions = dict(mo_energy=jnp.array([-.03, .02]), df_factors=base["df_factors"]*.05,
        fock_matrix=jnp.array([[.04, .01], [.01, -.025]]),
        hcore_matrix=jnp.array([[.03, -.01], [-.01, -.02]]),
        density_matrix=jnp.array([[.03, .01], [.01, -.02]]), eta=.005)
    def observable(t):
        arguments = dict(base)
        if leaf == "mo_coeff":
            angle = .04*t
            arguments[leaf] = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                                         [jnp.sin(angle), jnp.cos(angle)]])
        else:
            arguments[leaf] = base[leaf]+t*directions[leaf]
        result = g0w0_cd_restricted(**arguments, nocc=1, nw=8, qp_tol=1e-11)
        return jnp.vdot(result.mo_energy, jnp.array([1., .3]))
    point = jnp.array(0.)
    actual = jax.jit(jax.grad(observable))(point)
    h = 1e-4
    expected = (observable(point+h)-observable(point-h))/(2*h)
    assert abs(float(actual)) > 1e-10
    np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=3e-5)


@pytest.mark.parametrize("leaf", ["qp_energy", "screening_energy", "mo_factors"])
def test_bse_array_driver_preserves_physical_leaf_and_mask_response(leaf):
    from gradscf import bse

    base = _bse_inputs()
    space = bse.make_bse_space(3, 1)
    cfg = bse.BSEConfig(nroots=1, solver="dense", conv_tol=1e-11)
    inputs = (base.qp_energy, base.screening_energy, base.mo_factors)
    index = ("qp_energy", "screening_energy", "mo_factors").index(leaf)
    direction = (jnp.array([-.03, .02, .04]), jnp.array([.02, -.01, .03]), base.mo_factors*.1)[index]
    def observable(t):
        values = list(inputs)
        values[index] = values[index]+t*direction
        result = bse.run_bse(*values, space, config=cfg,
            qp_computed_mask=base.qp_computed_mask, qp_converged_mask=base.qp_converged_mask)
        amplitude = result.x_amplitudes
        return result.excitation_energies[0]+.2*jnp.sum(
            jnp.arange(amplitude.size).reshape(amplitude.shape)*amplitude**2)
    point = jnp.array(0.)
    actual = jax.jit(jax.grad(observable))(point)
    h = 1e-4
    np.testing.assert_allclose(actual, (observable(point+h)-observable(point-h))/(2*h),
                               atol=3e-8, rtol=3e-5)
    assert abs(float(actual)) > 1e-10
    solve = jax.jit(lambda mask: bse.run_bse(*inputs, space, config=cfg,
        qp_computed_mask=mask, qp_converged_mask=base.qp_converged_mask))
    assert bool(jnp.all(solve(base.qp_computed_mask).converged))
    invalid = solve(jnp.array([True, False, True]))
    assert not bool(jnp.any(invalid.converged))
    assert not bool(jnp.any(invalid.response_valid))
