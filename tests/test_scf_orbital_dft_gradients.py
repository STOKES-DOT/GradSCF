"""DFT SCF response on a deterministic finite AO quadrature (atomic units).

Three spatial orbitals, eight grid points, and fixed 2-alpha/1-beta occupations
keep both spin densities nonzero. These are derivative regressions, not a
molecular quadrature accuracy benchmark. Float64 is required by conftest.py.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax_xc")

from gradscf.scf.orbital_optimization import (
    minimize_gks_from_integrals,
    minimize_roks_from_integrals,
    minimize_uks_from_integrals,
)


def _solve(parameters, method, mode):
    rng = np.random.default_rng(913)
    ao_base = rng.normal(scale=.25, size=(8, 3))
    ao_base[:, 0] += 1.1
    ao_direction = rng.normal(scale=.12, size=(8, 3))
    deriv_base = rng.normal(scale=.18, size=(3, 8, 3))
    deriv_direction = rng.normal(scale=.08, size=(3, 8, 3))
    ao = jnp.asarray(ao_base) + parameters[1]*ao_direction
    deriv = jnp.concatenate((ao[None], deriv_base + parameters[1]*deriv_direction))
    weights = jnp.linspace(.035, .065, 8)*(1 + parameters[1]*jnp.linspace(-.3, .4, 8))
    h_direction = jnp.array([[.1, .2, -.12], [.2, -.08, .15], [-.12, .15, .04]])
    hcore = jnp.array([[-2.3, .08, -.03], [.08, -.7, .05], [-.03, .05, .9]])
    hcore = hcore + parameters[0]*h_direction
    factors = jnp.asarray([[[.2, .02, -.01], [.02, .14, .03], [-.01, .03, .12]]])
    eri = jnp.einsum("xpq,xrs->pqrs", factors, factors)
    angle = .19
    seed = np.array([[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    occ = np.array([[1., 1., 0.], [1., 0., 0.]])
    kwargs = dict(overlap=jnp.eye(3), hcore=hcore, eri=eri, nuclear_repulsion=.23,
                  ao=ao, ao_deriv1=deriv, grid_weights=weights,
                  xc_spec='lda_x' if method == 'gks_ncol' else 'pbe',
                  max_iterations=65, gradient_tolerance=1e-9, gradient_mode=mode)
    if method == 'uks':
        return minimize_uks_from_integrals(**kwargs, mo_coeff=np.stack([seed, seed]), mo_occ=occ)
    if method == 'roks':
        return minimize_roks_from_integrals(**kwargs, mo_coeff=seed, mo_occ=occ)
    spin = (np.array([[np.cos(.23), -1j*np.sin(.23)],
                      [-1j*np.sin(.23), np.cos(.23)]])
            if method == 'gks_ncol' else np.eye(2))
    return minimize_gks_from_integrals(**kwargs, mo_coeff=np.kron(spin, seed).astype(complex),
                                      mo_occ=occ.ravel(), collinear='ncol' if method == 'gks_ncol' else 'col')


def _observables(result):
    density = result.density_matrix
    spatial = density.sum(axis=0) if density.ndim == 3 else density[:3, :3] + density[3:, 3:]
    probe = jnp.array([[.13, .4, -.2], [.4, -.07, .3], [-.2, .3, .11]])
    return jnp.stack((result.total_energy, jnp.einsum('ij,ji->', probe, spatial).real))


@pytest.mark.parametrize('method', ['uks', 'roks', 'gks_col', 'gks_ncol'])
@pytest.mark.parametrize('mode', ['implicit', 'unrolled'])
def test_dft_energy_and_density_response_to_integrals_and_grid(method, mode):
    _check_response(method, mode, jnp.array([1., .3]))


def test_dft_density_only_implicit_response_to_integrals_and_grid():
    _check_response('uks', 'implicit', jnp.array([0., 1.]))


def _check_response(method, mode, observable_weights):
    parameters = jnp.array([.17, .09])
    solve = jax.jit(lambda p: _solve(p, method, mode))
    result = solve(parameters)
    assert bool(result.stationary), float(result.gradient_norm)
    assert float(result.gradient_norm) <= 1e-9
    fn = lambda p: jnp.dot(observable_weights, _observables(_solve(p, method, mode)))
    actual = jax.jit(jax.grad(fn))(parameters)
    step = 2e-5
    finite_difference = []
    for direction in jnp.eye(2):
        plus, minus = solve(parameters + step*direction), solve(parameters - step*direction)
        assert bool(plus.stationary) and bool(minus.stationary)
        finite_difference.append((_observables(plus) - _observables(minus))/(2*step))
    finite_difference = jnp.stack(finite_difference, axis=1)
    # Both columns must exercise the observable response, not only direct energy.
    assert np.all(np.abs(np.asarray(finite_difference[1])) > 1e-5)
    np.testing.assert_allclose(actual, observable_weights @ finite_difference, atol=2e-6, rtol=2e-5)
    other = jax.jit(lambda p: _solve(p, method, 'unrolled' if mode == 'implicit' else 'implicit'))(parameters)
    np.testing.assert_allclose(result.total_energy, other.total_energy, atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.density_matrix, other.density_matrix, atol=2e-11, rtol=0)
