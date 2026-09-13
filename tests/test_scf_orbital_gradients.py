"""SCF backward contracts: actual finite iterates and stationary response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.orbital_optimization import (
    minimize_uks_from_integrals, minimize_roks_from_integrals,
    minimize_gks_from_integrals,
)


def solve(theta, method, mode, *, max_iterations=50, varying_overlap=False):
    c = np.array([[np.cos(.37), -np.sin(.37)], [np.sin(.37), np.cos(.37)]])
    occ = np.array([[1., 0.], [0., 0.]])
    s = (jnp.array([[1., .1*theta], [.1*theta, 1.+.2*theta]])
         if varying_overlap else jnp.eye(2))
    h = jnp.array([[-1., theta], [theta, .5]])
    kwargs = dict(overlap=s, hcore=h, eri=jnp.zeros((2,)*4), nuclear_repulsion=0.,
        ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
        xc_spec='hf', max_iterations=max_iterations, gradient_tolerance=1e-9,
        gradient_mode=mode, orthonormalize_initial=varying_overlap)
    if method == 'uks':
        return minimize_uks_from_integrals(**kwargs, mo_coeff=np.stack([c,c]), mo_occ=occ)
    if method == 'roks':
        return minimize_roks_from_integrals(**kwargs, mo_coeff=c, mo_occ=occ)
    return minimize_gks_from_integrals(**kwargs,
        mo_coeff=np.kron(np.eye(2),c).astype(complex), mo_occ=occ.ravel())


def loss(theta, method, mode, **kwargs):
    result = solve(theta, method, mode, **kwargs)
    density = result.density_matrix
    if density.ndim == 3:
        density = density.sum(axis=0)
    return result.total_energy + .3*density[0, 1].real


@pytest.mark.parametrize('method', ['uks', 'roks', 'gks'])
@pytest.mark.parametrize('mode', ['implicit', 'unrolled'])
def test_orbital_energy_and_density_backward_matches_finite_difference(method, mode):
    theta, step = .17, 2e-5
    fn = lambda t: loss(t, method, mode)
    result = jax.jit(lambda t: solve(t, method, mode))(theta)
    assert result.stationary
    actual = jax.jit(jax.grad(fn))(theta)
    fd = (fn(theta+step)-fn(theta-step))/(2*step)
    np.testing.assert_allclose(actual, fd, atol=2e-7, rtol=0)
    other = solve(theta, method, 'unrolled' if mode=='implicit' else 'implicit')
    np.testing.assert_allclose(result.density_matrix, other.density_matrix, atol=1e-12, rtol=0)


@pytest.mark.parametrize('mode', ['implicit', 'unrolled'])
def test_overlap_response_includes_metric_constraint(mode):
    fn = lambda t: loss(t, 'uks', mode, varying_overlap=True)
    theta, step = .17, 2e-5
    actual = jax.grad(fn)(theta)
    fd = (fn(theta+step)-fn(theta-step))/(2*step)
    np.testing.assert_allclose(actual, fd, atol=3e-7, rtol=0)


def test_unrolled_differentiates_truncated_solver_and_implicit_rejects_it():
    fn = lambda t: loss(t, 'uks', 'unrolled', max_iterations=1)
    theta, step = .17, 2e-5
    assert not solve(theta, 'uks', 'unrolled', max_iterations=1).stationary
    fd = (fn(theta+step)-fn(theta-step))/(2*step)
    np.testing.assert_allclose(jax.grad(fn)(theta), fd, atol=2e-7, rtol=0)
    invalid = jax.grad(lambda t: loss(t, 'uks', 'implicit', max_iterations=1))(theta)
    assert not np.isfinite(invalid)


def test_implicit_backward_also_composes_outside_a_jitted_solve():
    fn = jax.jit(lambda theta: loss(theta,'uks','implicit'))
    theta, step = .17, 2e-5
    np.testing.assert_allclose(jax.grad(fn)(theta),
        (fn(theta+step)-fn(theta-step))/(2*step),atol=2e-7,rtol=0)
