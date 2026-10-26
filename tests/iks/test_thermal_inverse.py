"""Explicit finite-temperature matrix inversion plus local potential projection."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks
from gradscf.pbc.occupations import fermi_density
from test_potential import grid, functions


def entry():
    assert callable(getattr(iks, 'invert', None)), 'Missing explicit finite-basis inverse entry'
    return iks.invert


def problem(layout):
    s = jnp.array([[1.1, .05], [.05, .9]])
    basis = iks.Potential(grid(), s, functions, 2, params=.2, layout=layout)
    channels = {'rks': 1, 'uks': 2, 'gks': 4}[layout]
    b = jnp.arange(2*channels, dtype=float).reshape(channels, 2)*.02+.03
    if layout == 'rks': b = b[0]
    h = jnp.diag(jnp.array([-.2, .1]))
    if layout == 'uks': h = jnp.stack((h, h+.03*s))
    if layout == 'gks': h = jnp.kron(jnp.eye(2), h)
    f = h+basis.matrix(b)
    ss = jnp.kron(jnp.eye(2), s) if layout == 'gks' else s
    blocks = f if layout == 'uks' else f[None]
    counts = jnp.array([1., 1.]) if layout == 'uks' else jnp.array([1. if layout == 'rks' else 2.])
    d, occ = fermi_density(blocks[:, None], ss, counts, .2, jnp.ones(1))
    assert occ.converged
    d = d[:, 0] if layout == 'uks' else d[0, 0]*(2 if layout == 'rks' else 1)
    target = iks.Target(d, s, (1, 1) if layout == 'uks' else 2, layout=layout, energy=-.7)
    return target, basis, h


@pytest.mark.parametrize('layout', ['rks', 'uks', 'gks'])
def test_thermal_inverse_reproduces_manufactured_density(layout):
    solve = entry()
    target, basis, h = problem(layout)
    result = solve(target, basis, h, method='thermal_matrix', temperature=.2,
                   regularization=1e-12, tolerance=2e-7)
    assert result.valid and result.method == 'thermal_matrix'
    assert np.isfinite(result.fock).all() and np.isfinite(result.density).all()
    np.testing.assert_allclose(result.density, target.density, atol=2e-7, rtol=0., equal_nan=False)
    assert result.density_error < 2e-7 and result.operator_error < 2e-7


def test_thermal_inverse_does_not_clip_idempotent_targets():
    solve = entry()
    _, basis, h = problem('rks')
    target = iks.Target(2*jnp.linalg.inv(basis.overlap), basis.overlap, 4)
    with pytest.raises(ValueError, match='occupation|interior|thermal'):
        solve(target, basis, h, method='thermal_matrix', temperature=.2)


def test_unknown_inverse_method_does_not_fall_back_to_matrix_fit():
    solve = entry()
    target, basis, h = problem('rks')
    with pytest.raises(NotImplementedError, match='method|thermal_matrix'):
        solve(target, basis, h, method='wy', temperature=.2)


def test_ao_product_basis_factory_matches_explicit_products():
    assert callable(getattr(iks.Potential, 'ao_products', None)), 'Missing reusable local AO-product potential basis'
    provider = grid()
    basis = iks.Potential.ao_products(provider, jnp.eye(2), layout='rks')
    assert basis.nparam == 3
    coefficients = jnp.array([.2, -.1, .3])
    ao = provider.block(0, provider.ngrids)[0]
    values = coefficients[0]*ao[:, 0]**2+coefficients[1]*ao[:, 0]*ao[:, 1]+coefficients[2]*ao[:, 1]**2
    expected = jnp.einsum('g,gi,gj->ij', provider.weights*values, ao, ao)
    np.testing.assert_allclose(basis.matrix(coefficients), expected, atol=2e-13)


def test_inverse_is_invariant_to_potential_basis_rescaling():
    from dataclasses import replace
    target, basis, h = problem('uks')
    def rescaled(params, coordinates):
        return functions(params, coordinates)*jnp.array([1e-6, 1e3])
    changed = replace(basis, basis=rescaled)
    result = entry()(target, changed, h, method='thermal_matrix', temperature=.2,
                     regularization=1e-12, tolerance=2e-7)
    assert result.valid
    np.testing.assert_allclose(result.density, target.density, atol=2e-7, rtol=0)


def test_reference_generation_rejects_unimplemented_inverse_map_ad():
    target, basis, h = problem('rks')
    with pytest.raises(NotImplementedError, match='reference|offline|inverse.map'):
        jax.grad(lambda t: jnp.sum(entry()(target, basis, h, method='thermal_matrix',
            temperature=t).density))(.2)
