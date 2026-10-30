"""Offline Wu-Yang density inversion with explicit reference validity."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks
from gradscf.solvers.eigen.spectral import inverse_sqrt
from test_potential import grid, functions


def manufactured(layout='rks', basis_function=functions, nparam=2):
    s = jnp.array([[1.1, .05], [.05, .9]])
    basis = iks.Potential(grid(), s, basis_function, nparam, params=.2, layout=layout)
    base = jnp.diag(jnp.array([-.4, .3]))
    b = jnp.array([.12, -.08])
    if layout == 'uks':
        base = jnp.stack((base, base+.02*s))
        b = jnp.stack((b, -.7*b))
    source = iks.Potential(grid(), s, functions, 2, params=.2, layout=layout)
    f = base+source.matrix(b)
    x = inverse_sqrt(s, 1e-12)
    _, u = jnp.linalg.eigh(x@f@x)
    c = x@u
    occ = jnp.array([2., 0.]) if layout == 'rks' else jnp.array([[1., 0.], [1., 0.]])
    d = (c*occ[..., None, :])@c.swapaxes(-1, -2)
    target = iks.Target(d, s, (1, 1) if layout == 'uks' else 2, layout=layout)
    return target, basis, base


@pytest.mark.parametrize('layout', ['rks', 'uks'])
def test_wu_yang_recovers_density_and_complete_spectral_reference(layout):
    target, basis, base = manufactured(layout)
    result = iks.invert(target, basis, base, method='wu_yang', regularization=0.,
                        tolerance=1e-7, gradient_tolerance=1e-10, maxiter=300)
    assert result.valid and result.converged
    assert result.method == 'wu_yang' and result.reference.method == 'wu_yang'
    assert result.reference.valid and result.reference.integer_occupations
    assert result.gap > 0 and result.moment_error < 1e-7 and result.density_error < 1e-7
    np.testing.assert_allclose(result.density, target.density, atol=2e-6, rtol=0.)
    s = target.overlap
    c = result.reference.coefficients
    np.testing.assert_allclose(c.swapaxes(-1, -2)@s@c, np.broadcast_to(np.eye(2), c.shape), atol=1e-10)
    np.testing.assert_allclose(result.fock@c, (s@c)*result.reference.energies[..., None, :], atol=1e-10)
    np.testing.assert_allclose(base+basis.matrix(result.potential_coefficients, constant=result.constant),
                               result.fock, atol=1e-11)


def test_wu_yang_redundant_rescaled_potential_basis_is_rank_reduced():
    def redundant(params, coords):
        f = functions(params, coords)
        return jnp.stack((f[:, 0], 1e6*f[:, 0], 1e-5*f[:, 1]), axis=-1)
    target, basis, base = manufactured(basis_function=redundant, nparam=3)
    result = iks.invert(target, basis, base, method='wu_yang', regularization=0.,
                        tolerance=1e-7, gradient_tolerance=1e-10)
    assert result.valid and result.projection_rank == 2
    np.testing.assert_allclose(result.density, target.density, atol=2e-6, rtol=0.)


def test_wu_yang_reports_iteration_failure_instead_of_a_valid_teacher():
    target, basis, base = manufactured()
    result = iks.invert(target, basis, base, method='wu_yang', maxiter=1,
                        regularization=0., tolerance=1e-12, gradient_tolerance=1e-12)
    assert not result.valid and not result.converged
    assert result.moment_error > 1e-12
    assert result.reference is None


def test_wu_yang_rejects_inverse_map_ad_and_unsupported_temperature():
    target, basis, base = manufactured()
    with pytest.raises(NotImplementedError, match='offline|inverse.map|AD'):
        jax.grad(lambda t: iks.invert(target, basis, base+t*jnp.eye(2),
                                    method='wu_yang').density.sum())(.1)
    with pytest.raises((TypeError, ValueError), match='temperature'):
        iks.invert(target, basis, base, method='wu_yang', temperature=.1)


def test_wu_yang_objective_gradient_and_hvp_match_finite_differences(monkeypatch):
    from gradscf.dft.iks import local as wu_yang
    original = wu_yang.minimize_eager
    checked = []

    def inspect(value_grad, initial, **kwargs):
        point = initial+jnp.array([.04, -.02])
        direction = jnp.array([.3, -.4])
        step = 1e-5
        value, gradient = value_grad(point)
        numeric = (value_grad(point+step*direction)[0]-value_grad(point-step*direction)[0])/(2*step)
        np.testing.assert_allclose(jnp.dot(gradient, direction), numeric, atol=2e-10, rtol=2e-7)
        hvp = jax.jvp(lambda z: value_grad(z)[1], (point,), (direction,))[1]
        numeric_hvp = (value_grad(point+step*direction)[1]-value_grad(point-step*direction)[1])/(2*step)
        np.testing.assert_allclose(hvp, numeric_hvp, atol=2e-9, rtol=2e-6)
        checked.append(True)
        return original(value_grad, initial, **kwargs)

    monkeypatch.setattr(wu_yang, 'minimize_eager', inspect)
    target, basis, base = manufactured()
    result = iks.invert(target, basis, base, method='wu_yang', regularization=0.,
                        tolerance=1e-7, gradient_tolerance=1e-10)
    assert checked and result.valid


def test_wu_yang_removes_constant_gauge_before_column_normalization():
    provider = grid()
    ao = provider.block(0, provider.ngrids)[0]
    s = jnp.einsum('g,gi,gj->ij', provider.weights, ao, ao)
    x = inverse_sqrt(s, 1e-12)
    base = jnp.diag(jnp.array([-.4, .3]))
    _, u = jnp.linalg.eigh(x@base@x)
    occupied = (x@u)[:, :1]
    target = iks.Target(2*occupied@occupied.T, s, 2)
    basis = iks.Potential(provider, s, lambda _, coords: jnp.ones((coords.shape[0], 1)), 1)
    with pytest.raises(ValueError, match='observable|gauge'):
        iks.invert(target, basis, base, method='wu_yang')


def test_density_acceptance_metric_is_explicit_and_reports_absolute_and_relative_norms():
    target, basis, base = manufactured()
    result = iks.invert(target, basis, base, method='wu_yang', regularization=1e-4,
                        tolerance=.1, gradient_tolerance=1e-9,
                        density_metric='relative_l2')
    ao = basis.grid.block(0, basis.grid.ngrids)[0]
    target_rho = jnp.einsum('gi,ij,gj->g', ao, target.density, ao)
    reference_norm = jnp.sqrt(jnp.sum(basis.grid.weights*target_rho**2))
    np.testing.assert_allclose(result.relative_density_error,
                               result.density_error/reference_norm, atol=1e-12)
    assert result.density_metric == 'relative_l2'
    with pytest.raises(ValueError, match='density_metric'):
        iks.invert(target, basis, base, method='wu_yang', density_metric='unknown')


def test_local_inverse_projects_only_roundoff_antisymmetry_before_spectral_audit():
    target,basis,base = manufactured()
    noisy = base+1e-13*jnp.array([[0.,1.],[-1.,0.]])
    result = iks.invert(target,basis,noisy,method='wu_yang',regularization=0.,
                        tolerance=1e-6,gradient_tolerance=1e-9)
    assert result.valid
    assert result.spectral_converged and result.eigen_residual_max < 1e-9
    np.testing.assert_allclose(result.fock,result.fock.T,atol=1e-15,rtol=0.)
