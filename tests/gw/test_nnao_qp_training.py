"""Synthetic radial weights -> native RI -> implicit RHF -> nonlinear QP loss.

Float64 and Hartree; H2 is fixed at 0.74 Angstrom. Four tiny model outputs
control one log-exponent scale and one relative contraction per H center.
This exercises the NNAO radial-parameter interface, not a trained MACE model,
core masking (H has no core), dataset accuracy, or geometry/auxiliary AD.
The native 2c auxiliary metric is fixed; native contracted 3c data, S and h
are rebuilt for every parameter value, without a primitive four-index ERI.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np
import optax
import pytest

from gradscf import integrals, scf
from gradscf.gw import g0w0_cd_restricted
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.scf.core import _diagonalize_fock, _restricted_density_from_fock
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
from gradscf.solvers.nonlinear import (
    ImplicitFixedPointConfig,
    implicit_fixed_point_solution,
)


def _problem():
    atom = "H 0 0 0; H 0 0 .74"
    top, parameters = integrals.prepare_basis(
        atom, {"H": [[0, [1.2, .7], [.28, .35]]]}, cart=False)
    auxiliary, auxiliary_parameters = integrals.prepare_basis(
        atom, "def2-universal-jkfit", cart=False)
    plan = integrals.make_plan(top)
    ri = integrals.make_auxiliary_plan(top, auxiliary)
    metric = ri.metric_factor(parameters, auxiliary_parameters)
    enuc = scf.nuclear_repulsion_energy(
        parameters.nuclear_coords, jnp.asarray(top.nuclear_charges))
    config = RKSConfig(xc_spec="hf", jk_backend="df", max_cycle=150,
                       conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)
    identity = jnp.eye(top.nao, dtype=jnp.float64)

    def bind(weights):
        return replace(parameters,
            exponents=tuple(alpha * jnp.exp(weights[i])
                            for i, alpha in enumerate(parameters.exponents)),
            coefficients=tuple(c.at[1, 0].set(weights[2 + i])
                               for i, c in enumerate(parameters.coefficients)))

    def fixed_point(density, physical):
        core, factors = physical
        coulomb, exchange = build_jk_from_df(factors, density)
        return _restricted_density_from_fock(core + coulomb - .5 * exchange,
                                             identity, 1)

    def loss(weights):
        bound = bind(weights)
        overlap = plan.evaluate("overlap", bound)
        hcore = plan.evaluate("kinetic", bound) + plan.evaluate("nuclear", bound)
        factors = unpack_factors(ri.factors(bound, auxiliary_parameters,
                                            metric_factor=metric), top.nao)
        # Match the NNAO DF-RHF objective's invertible Cholesky density state.
        # The complete overlap response remains in transform, h and b.
        lower = jnp.linalg.cholesky(overlap)
        transform = solve_triangular(lower.T, identity, lower=False)
        h = transform.T @ hcore @ transform
        b = transform.T @ factors @ transform
        result = run_rks_from_integrals_traceable(
            overlap=identity, hcore=h, eri=None, df_factors=b, nelectron=2,
            nuclear_repulsion=enuc, ao=jnp.zeros((0, top.nao)),
            ao_deriv1=jnp.zeros((4, 0, top.nao)), grid_weights=jnp.zeros(0),
            config=config)
        density = implicit_fixed_point_solution(
            (h, b), solution=result.density_matrix, fixed_point=fixed_point,
            config=ImplicitFixedPointConfig(tolerance=1e-10, max_iter=100, restart=4),
            converged=result.converged, require_converged=True)
        coulomb, exchange = build_jk_from_df(b, density)
        fock = h + coulomb - .5 * exchange
        energy, coefficients = _diagonalize_fock(fock, identity)
        ao_density = transform @ density @ transform.T
        ao_coefficients = transform @ coefficients
        # Rebuild AO Fock from the same responding density used by GW.
        coulomb, exchange = build_jk_from_df(factors, ao_density)
        ao_fock = hcore + coulomb - .5 * exchange
        qp = g0w0_cd_restricted(
            mo_energy=energy, mo_coeff=ao_coefficients, nocc=1,
            df_factors=factors, fock_matrix=ao_fock, hcore_matrix=hcore,
            density_matrix=ao_density, nw=8, eta=.03, orbs=(0,),
            g_orbitals=(0, 1), screening_occupied=(0,), screening_virtual=(1,),
            qp_solver="newton", qp_tol=1e-11, diff_mode="implicit", max_memory=64.)
        info = dict(scf_converged=result.converged, qp_converged=qp.converged,
            scf_residual=jnp.linalg.norm(fixed_point(density, (h, b)) - density),
            qp_residual=qp.qp_residual[0], qp_energy=qp.mo_energy[0],
            overlap=overlap, factors=factors, density=ao_density,
            mo_energy=energy, mo_coeff=ao_coefficients)
        # A synthetic fixed target tests training mechanics, not QP accuracy.
        return .5 * (qp.mo_energy[0] + .3) ** 2, info

    # Small center asymmetry makes the self-consistent density response active.
    weights = jnp.array([.03, -.02, .35, .32], dtype=jnp.float64)
    return loss, weights, bind, ri, metric, auxiliary_parameters


def _check(value, info):
    assert bool(info["scf_converged"]) and bool(info["qp_converged"])
    assert float(info["scf_residual"]) < 1e-8
    assert abs(float(info["qp_residual"])) < 1e-10
    assert np.isfinite(value)
    for key in ("factors", "density", "mo_energy", "mo_coeff", "qp_energy"):
        assert np.isfinite(info[key]).all(), key
    np.testing.assert_allclose(info["mo_coeff"].T @ info["overlap"] @ info["mo_coeff"],
                               np.eye(2), atol=2e-10, rtol=0.)


def test_nnao_radial_qp_loss_gradient_update_and_parameter_rebinding():
    loss, weights, bind, ri, metric, auxiliary_parameters = _problem()
    evaluate = jax.jit(jax.value_and_grad(loss, has_aux=True))
    primal = jax.jit(loss)
    (value, info), gradient = evaluate(weights)
    _check(value, info)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient[:2]) > 1e-8  # log-exponents
    assert np.linalg.norm(gradient[2:]) > 1e-8  # relative contractions

    # Both displaced calls rebuild integrals and reconverge SCF and QP roots.
    def checked(point):
        value, info = primal(point)
        _check(value, info)
        return value

    for step in (2e-4, 1e-4):
        finite = jnp.stack([(checked(weights + step * direction)
                             - checked(weights - step * direction)) / (2 * step)
                            for direction in jnp.eye(4)])
        np.testing.assert_allclose(gradient, finite, atol=3e-7, rtol=3e-5)

    # Same compiled loss/backward: changed radial values cannot reuse stale
    # contracted integrals, mean-field orbitals or screened self energies.
    changed = weights + jnp.array([.01, -.02, .005, -.007])
    (changed_value, changed_info), changed_gradient = evaluate(changed)
    _check(changed_value, changed_info)
    assert np.isfinite(changed_gradient).all()
    for key in ("factors", "density", "mo_energy", "qp_energy"):
        assert np.linalg.norm(np.asarray(changed_info[key] - info[key])) > 1e-8, key
    assert abs(float(changed_value - value)) > 1e-9
    (repeated, repeated_info), repeated_gradient = evaluate(weights)
    _check(repeated, repeated_info)
    np.testing.assert_allclose(repeated, value, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(repeated_gradient, gradient, atol=1e-11, rtol=1e-9)

    optimizer = optax.adam(1e-3)
    updates, state = optimizer.update(gradient, optimizer.init(weights), weights)
    updated = optax.apply_updates(weights, updates)
    (updated_value, updated_info), updated_gradient = evaluate(updated)
    _check(updated_value, updated_info)
    assert np.isfinite(updated_gradient).all()
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(state))
    assert np.linalg.norm(np.asarray(updated - weights)) > 0.
    assert np.all(np.concatenate(bind(updated).exponents) > 0.)

    # The auxiliary metric is valid across radial changes, but its signature
    # must reject a geometry change. This is not a geometry derivative test.
    moved = replace(bind(weights),
                    nuclear_coords=bind(weights).nuclear_coords.at[1, 2].add(.01))
    with pytest.raises(ValueError, match="Cached auxiliary metric signature"):
        ri.factors(moved, auxiliary_parameters, metric_factor=metric)
