"""Restricted full-TDDFT first-order vector AD, without native integrals.

All numbers are float64; frequencies/Fock entries are in Hartree.  The toy
reference is deliberately noncanonical and its XC kernel has a dynamic bound
parameter, so an energy-only derivative cannot satisfy the amplitude checks.
"""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule
from gradscf.tddft import RestrictedCasidaTDDFT, gen_tdhf_vind
from gradscf.tddft.casida import solve_casida_from_tdhf_operator


@dataclass(frozen=True)
class _LocalKernel:
    strength: object = 0.2
    exact_exchange_fraction: float = 0.25

    def bind(self, parameters):
        return replace(self, strength=parameters)

    def local_kernel(self, density):
        return self.strength * (0.03 + 0.02 * density)


def _reference():
    rng = np.random.default_rng(804)
    factors = rng.normal(size=(4, 5, 5)) * 0.07
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("Ppq,Prs->pqrs", factors, factors)
    ao = rng.normal(size=(7, 5)) * 0.3
    dipole = rng.normal(size=(3, 5, 5)) * 0.2
    dipole = (dipole + dipole.transpose(0, 2, 1)) / 2
    energies = jnp.array([-1.3, -0.7, 0.2, 0.6, 1.1], dtype=jnp.float64)
    fock = jnp.diag(energies).at[0, 1].set(0.035).at[1, 0].set(0.035)
    fock = fock.at[2, 3].set(-0.045).at[3, 2].set(-0.045)
    fock = fock.at[3, 4].set(0.025).at[4, 3].set(0.025)
    return RestrictedMolecule(
        ao=jnp.asarray(ao), grid=QuadratureGrid(jnp.linspace(0.1, 0.3, 7)),
        dipole_integrals=jnp.asarray(dipole), rep_tensor=jnp.asarray(eri),
        mo_coeff=jnp.eye(5), mo_occ=jnp.array([2., 2., 0., 0., 0.]),
        mo_energy=energies,
        rdm1=jnp.stack([jnp.diag(jnp.array([1., 1., 0., 0., 0.]))] * 2),
        h1e=fock, fock_matrix=fock, nuclear_repulsion=0., nocc=2,
    )


def _displaced_reference(reference, parameters):
    # Independent occupied/virtual Fock changes affect the complete A action.
    dfock = jnp.array([[0., .04, 0., 0., 0.], [.04, -.02, 0., 0., 0.],
                      [0., 0., .03, .08, -.02], [0., 0., .08, -.01, .05],
                      [0., 0., -.02, .05, .02]], dtype=jnp.float64)
    return replace(reference, fock_matrix=reference.fock_matrix + parameters[0] * dfock)


def _solve(reference, parameters, *, mode="implicit_eigenvector", maxiter=120):
    return RestrictedCasidaTDDFT(
        _displaced_reference(reference, parameters), _LocalKernel(),
        xc_params=parameters[1], tda_gradient_mode=mode,
        davidson_tol=1e-10, davidson_max_iter=maxiter,
        davidson_max_subspace=5 if maxiter == 1 else 6, eigenvector_adjoint_tol=1e-11,
        eigenvector_adjoint_max_iter=100,
        response_kernel_options={"two_electron_mode": "direct", "integral_cache": "none"},
    ).kernel(nstates=2)


def _phase(result, anchor):
    x = result.x_amplitudes.reshape(2, -1)
    y = result.y_amplitudes.reshape(2, -1)
    overlap = jnp.sum(x * anchor.x_amplitudes.reshape(2, -1)
                      - y * anchor.y_amplitudes.reshape(2, -1), axis=1)
    signs = jax.lax.stop_gradient(jnp.where(overlap < 0, -1., 1.))
    return x * signs[:, None], y * signs[:, None]


def test_full_driver_vector_jvp_vjp_and_xc_parameter_match_central_fd():
    reference = _reference()
    parameters = jnp.array([0.13, 0.27])
    anchor = _solve(reference, parameters)
    assert bool(anchor.converged)

    def observable(p):
        result = _solve(reference, p)
        x, y = _phase(result, anchor)
        # X and Y are separate outputs: X+Y alone can hide sign/metric errors.
        return jnp.concatenate([x.reshape(-1), y.reshape(-1)])

    inner = jax.jit(observable)
    direction = jnp.array([0.4, -0.7])
    cotangent = jnp.linspace(-0.3, 0.7, 24)
    tangent = jax.jit(lambda p: jax.jvp(inner, (p,), (direction,))[1])(parameters)
    jacobian = jax.jit(jax.jacfwd(inner))(parameters)
    reverse = jax.jit(jax.grad(lambda p: jnp.vdot(cotangent, inner(p))))(parameters)
    _, pullback = jax.vjp(inner, parameters)
    np.testing.assert_allclose(pullback(cotangent)[0], reverse, atol=2e-10, rtol=0)
    finite = (inner(parameters + 1e-4 * direction)
              - inner(parameters - 1e-4 * direction)) / 2e-4
    # Both physical inputs must reach amplitudes, independently of energies.
    assert np.linalg.norm(np.asarray(jacobian[:12, 0])) > 1e-4
    assert np.linalg.norm(np.asarray(jacobian[12:, 1])) > 1e-7
    np.testing.assert_allclose(tangent, finite, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(jnp.vdot(reverse, direction), jnp.vdot(cotangent, tangent),
                               atol=2e-10, rtol=0)


def test_full_vector_mode_preserves_forward_energies_and_metric():
    reference = _reference()
    parameters = jnp.array([0.13, 0.27])
    implicit = _solve(reference, parameters)
    legacy = _solve(reference, parameters, mode="eigenvalue_only")
    assert bool(implicit.converged) and bool(legacy.converged)
    np.testing.assert_allclose(implicit.excitation_energies, legacy.excitation_energies,
                               atol=3e-10, rtol=0)
    x, y = _phase(implicit, legacy)
    np.testing.assert_allclose(x, legacy.x_amplitudes.reshape(2, -1), atol=2e-9, rtol=0)
    np.testing.assert_allclose(y, legacy.y_amplitudes.reshape(2, -1), atol=2e-9, rtol=0)
    np.testing.assert_allclose(2 * (x @ x.T - y @ y.T), np.eye(2), atol=2e-9, rtol=0)
    old_tangent = jax.jvp(lambda p: _solve(reference, p, mode="eigenvalue_only").x_amplitudes,
                          (parameters,), (jnp.ones_like(parameters),))[1]
    np.testing.assert_array_equal(old_tangent, 0.)


def test_full_amplitudes_match_independent_nonsymmetric_dense_eigenproblem():
    reference = _reference()
    parameters = jnp.array([0.13, 0.27])
    result = _solve(reference, parameters)
    physical = _displaced_reference(reference, parameters)
    # This test-only oracle materializes the 12x12 TDHF matrix; production must
    # retain the existing microbatched A/B action and matrix-free response solve.
    matrix = np.asarray(gen_tdhf_vind(physical, _LocalKernel(), xc_params=parameters[1])(
        jnp.eye(12))).T
    values, vectors = np.linalg.eig(matrix)
    selected = np.argsort(np.where((values.real > 1e-7) & (abs(values.imag) < 1e-10),
                                  values.real, np.inf))[:2]
    z = vectors[:, selected].real
    z *= np.sqrt(0.5 / np.sum(z[:6] ** 2 - z[6:] ** 2, axis=0))[None, :]
    actual = np.concatenate([np.asarray(result.x_amplitudes).reshape(2, -1).T,
                             np.asarray(result.y_amplitudes).reshape(2, -1).T])
    z *= np.where(np.sum(z * actual, axis=0) < 0, -1., 1.)[None, :]
    np.testing.assert_allclose(result.excitation_energies, values[selected].real,
                               atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual, z, atol=2e-9, rtol=0)


def test_full_vector_nonconvergence_cannot_return_a_zero_derivative():
    reference = _reference()
    parameters = jnp.array([0.13, 0.27])

    def loss(p):
        result = _solve(reference, p, maxiter=1)
        return jnp.sum(result.x_amplitudes ** 4 + result.y_amplitudes ** 4)

    inner = jax.jit(loss)
    forward = jax.jit(lambda p: jax.jvp(inner, (p,), (jnp.ones_like(p),))[1])(parameters)
    reverse = jax.jit(jax.grad(inner))(parameters)
    assert not np.isfinite(forward)
    assert not np.isfinite(reverse).all()


def test_full_driver_rejects_invalid_gradient_mode():
    with pytest.raises(ValueError, match="gradient mode"):
        _solve(_reference(), jnp.array([0.13, 0.27]), mode="unsupported")


def _array_solve(a, b):
    dim = a.shape[0]

    def rows_action(rows):
        assert rows.shape[0] <= 8, "Full physical identity was requested."
        x, y = rows[:, :dim], rows[:, dim:]
        return jnp.concatenate([x @ a.T + y @ b.T, -x @ b.T - y @ a.T], axis=1)

    return solve_casida_from_tdhf_operator(
        jnp.diag(a)[None, :], rows_action, nstates=2,
        tda_gradient_mode="implicit_eigenvector", davidson_tol=1e-10,
        davidson_max_subspace=8, eigenvector_adjoint_tol=1e-11,
        eigenvector_adjoint_max_iter=100,
    )


def test_full_array_operator_retains_closed_over_a_and_b_under_nested_jit():
    rng = np.random.default_rng(405)
    u = jnp.asarray(rng.normal(size=(12, 3)) * 0.03)
    a = jnp.diag(jnp.linspace(0.7, 2.8, 12)) + u @ u.T
    b = jnp.eye(12) * 0.02 + 0.4 * u @ u.T
    da = jnp.asarray(rng.normal(size=(12, 12)) * 0.02)
    da = (da + da.T) / 2
    db = jnp.asarray(rng.normal(size=(12, 12)) * 0.008)
    db = (db + db.T) / 2
    probe = jnp.linspace(-0.4, 0.5, 12)

    def observable(parameters):
        result = _array_solve(a + parameters[0] * da, b + parameters[1] * db)
        x = result.x_amplitudes.reshape(2, 12)
        y = result.y_amplitudes.reshape(2, 12)
        return jnp.concatenate([(x @ probe) ** 2, (y @ probe) ** 2])

    parameters, direction = jnp.array([0.1, -0.07]), jnp.array([0.4, -0.8])
    inner = jax.jit(observable)
    cotangent = jnp.array([0.2, 0.5, -0.3, 0.7])
    forward = jax.jit(lambda p: jax.jvp(inner, (p,), (direction,))[1])(parameters)
    reverse = jax.jit(jax.grad(lambda p: jnp.vdot(cotangent, inner(p))))(parameters)
    finite = (inner(parameters + 1e-4 * direction)
              - inner(parameters - 1e-4 * direction)) / 2e-4
    assert np.linalg.norm(np.asarray(forward[:2])) > 1e-6
    assert np.linalg.norm(np.asarray(forward[2:])) > 1e-9
    np.testing.assert_allclose(forward, finite, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(jnp.vdot(reverse, direction), jnp.vdot(cotangent, forward),
                               atol=2e-10, rtol=0)


def test_full_array_degenerate_root_guard_invalidates_zero_physical_response():
    a = jnp.diag(jnp.array([0.8, 0.8, 1.5, 1.9, 2.3, 2.8]))
    b = jnp.zeros_like(a)

    def loss(t):
        result = _array_solve(t * a, b)
        return jnp.sum(result.x_amplitudes ** 4 + result.y_amplitudes ** 4)

    inner = jax.jit(loss)
    forward = jax.jit(lambda t: jax.jvp(inner, (t,), (jnp.ones_like(t),))[1])(1.)
    reverse = jax.jit(jax.grad(inner))(1.)
    assert not np.isfinite(forward)
    assert not np.isfinite(reverse)


def test_full_custom_xc_mlp_parameters_keep_x_and_y_response():
    from gradscf import dft

    reference = _reference()
    weights = jnp.linspace(0.1, 0.4, 15).reshape(5, 3)
    dweights = 0.1 * jnp.cos(jnp.arange(15.)).reshape(5, 3)

    def energy(params, density):
        hidden = jnp.tanh(density @ params["weight"])
        return params["scale"] * jnp.sum(hidden ** 2)

    # Functional differentiates this global density-matrix energy to obtain
    # its full nonlocal AO Hessian action, rather than the legacy local kernel.
    xc = dft.custom_xc(input_fn=lambda state: state.density_matrix,
                       energy_fn=energy, name="full-vector-mlp-response")

    def solve(p):
        return RestrictedCasidaTDDFT(
            reference, xc, xc_params={"weight": weights + p[1] * dweights, "scale": p[0]},
            tda_gradient_mode="implicit_eigenvector", davidson_tol=1e-10,
            davidson_max_iter=120, davidson_max_subspace=6,
            eigenvector_adjoint_tol=1e-11, eigenvector_adjoint_max_iter=100,
            response_kernel_options={"two_electron_mode": "direct", "integral_cache": "none"},
        ).kernel(nstates=2)

    point, direction = jnp.array([0.003, 0.1]), jnp.array([0.2, -0.7])
    anchor = solve(point)
    assert bool(anchor.converged)

    def observable(p):
        x, y = _phase(solve(p), anchor)
        return jnp.concatenate([x.reshape(-1), y.reshape(-1)])

    inner = jax.jit(observable)
    tangent = jax.jit(lambda p: jax.jvp(inner, (p,), (direction,))[1])(point)
    cotangent = jnp.linspace(-0.4, 0.6, 24)
    reverse = jax.jit(lambda p: jax.vjp(inner, p)[1](cotangent)[0])(point)
    finite = (inner(point + 1e-4 * direction) - inner(point - 1e-4 * direction)) / 2e-4
    assert np.linalg.norm(np.asarray(tangent[:12])) > 1e-6
    assert np.linalg.norm(np.asarray(tangent[12:])) > 1e-6
    assert np.linalg.norm(np.asarray(inner(point + jnp.array([0., 0.01])) - inner(point))) > 1e-9
    np.testing.assert_allclose(tangent, finite, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(jnp.vdot(reverse, direction), jnp.vdot(cotangent, tangent),
                               atol=2e-10, rtol=0)
