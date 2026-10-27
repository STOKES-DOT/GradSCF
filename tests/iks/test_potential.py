"""Block-local potential maps, physical transpose and analytic constant gauge."""
from dataclasses import replace
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft
from gradscf.integrals.grids.blocks import AOGrid


def api():
    assert hasattr(dft, 'iks'), 'Missing public gradscf.dft.iks infrastructure'
    return dft.iks


def functions(params, coords):
    return jnp.stack((coords[:, 0]+params, coords[:, 2]**2+.1), axis=-1)


def grid():
    return AOGrid(jnp.array([[.1*i, .2, .07*i] for i in range(7)]),
        jnp.linspace(.1, .3, 7), jnp.array([[0., 0., 0.], [.5, 0., .1]]),
        jnp.array([[.7], [1.1]]), jnp.ones((2, 1)), (1, 1), ((0, 0, 0),)*2, 3)


def expected_matrices(layout, panels):
    if layout == 'rks': return panels[0]
    if layout == 'uks': return panels
    a, x, y, z = panels
    return jnp.block([[a+z, x-1j*y], [x+1j*y, a-z]])


@pytest.mark.parametrize('layout,channels', [('rks', 1), ('uks', 2), ('gks', 4)])
def test_grid_map_dense_oracle_transpose_and_joint_hvp(layout, channels):
    module = api()
    provider = grid()
    overlap = jnp.array([[1.3, .1], [.1, .9]])
    coefficients = jnp.arange(channels*2, dtype=jnp.float64).reshape(channels, 2)*.07+.1
    coefficients = coefficients[0] if layout == 'rks' else coefficients
    shape = (2, 2) if layout == 'rks' else ((2, 2, 2) if layout == 'uks' else (4, 4))
    probe = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)*.03
    probe = (probe+probe.swapaxes(-1, -2))/2
    if layout == 'gks': probe = probe.astype(complex).at[0, 2].add(.12j).at[2, 0].add(-.12j)

    def evaluate(t, dense):
        moved = replace(provider, coords=provider.coords+t*.02, weights=provider.weights*(1+t*.01))
        values = coefficients*(1+t*.03)
        basis = module.Potential(moved, overlap, functions, 2, params=.2+t*.05,
                                 layout=layout, max_memory=1.)
        actual = basis.matrix(values)
        if dense:
            ao = moved.block(0, moved.ngrids)[0]
            feature = functions(.2+t*.05, moved.coords)
            potential = jnp.atleast_2d(values)@feature.T
            panels = jnp.einsum('g,gm,gn,cg->cmn', moved.weights, ao, ao, potential)
            actual = expected_matrices(layout, panels)
        return jnp.einsum('...ij,...ji->', actual, probe).real

    bundles = []
    for dense in (False, True):
        bundles.append(jax.jit(lambda t: jax.jvp(jax.value_and_grad(lambda x: evaluate(x, dense)),
            (t,), (jnp.ones_like(t),)))(jnp.asarray(.01)))
    for a, b in zip(jax.tree.leaves(bundles[0]), jax.tree.leaves(bundles[1])):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12, equal_nan=False)
    basis = module.Potential(provider, overlap, functions, 2, params=.2, layout=layout)
    lhs = jnp.einsum('...ij,...ji->', basis.matrix(coefficients), probe).real
    rhs = jnp.sum(coefficients*basis.moments(probe))
    np.testing.assert_allclose(lhs, rhs, atol=2e-12)
    adjoint = jax.grad(lambda b: jnp.einsum('...ij,...ji->', basis.matrix(b), probe).real)(coefficients)
    np.testing.assert_allclose(adjoint, basis.moments(probe), atol=2e-12)
    if os.environ.get('GRADSCF_TEST_GPU') == '1':
        assert jax.default_backend() == 'gpu'
        assert adjoint.devices() == {jax.devices('gpu')[0]}
    assert basis.matrix(coefficients).shape == shape


@pytest.mark.parametrize('layout,shift', [('rks', .3), ('uks', [.3, -.2]), ('gks', .3)])
def test_constant_gauge_uses_overlap_and_keeps_magnetic_fields(layout, shift):
    module = api()
    overlap = jnp.array([[1.2, .1], [.1, .8]])
    actual = module.constant_potential(overlap, shift, layout=layout)
    if layout == 'rks': expected = .3*overlap
    elif layout == 'uks': expected = jnp.stack((.3*overlap, -.2*overlap))
    else: expected = jnp.kron(jnp.eye(2), .3*overlap)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    provider = grid()
    basis = module.Potential(provider, overlap, functions, 2, params=.2, layout=layout)
    channels = {'rks': 1, 'uks': 2, 'gks': 4}[layout]
    coeff = jnp.zeros((channels, 2))
    coeff = coeff[0] if layout == 'rks' else coeff
    np.testing.assert_allclose(basis.matrix(coeff, constant=shift), expected, atol=1e-14)
    if layout == 'gks':
        with pytest.raises(ValueError, match='scalar|constant'):
            module.constant_potential(overlap, jnp.ones(4), layout='gks')


@pytest.mark.parametrize('budget', [1e-6, .001])
def test_tight_budget_rejects_before_evaluating_basis_functions(budget):
    def forbidden(*args): pytest.fail('Budget rejection occurred after potential evaluation')
    with pytest.raises(ValueError, match='memory|budget|resident'):
        api().Potential(grid(), jnp.eye(2), forbidden, 2, max_memory=budget).matrix(jnp.ones(2))


def test_memory_tiling_preserves_grid_and_basis_parameter_derivatives():
    module = api()
    provider = replace(grid(), block_size=7)
    loose = module.Potential(provider, jnp.eye(2), functions, 2, params=.2, max_memory=1.)
    widths = []
    def observed(params, coordinates):
        widths.append(coordinates.shape[0])
        return functions(params, coordinates)
    tight = replace(loose, max_memory=.003, basis=observed)
    a, b = (jax.jit(lambda p: jax.value_and_grad(lambda x: jnp.sum(
        replace(v, params=x).matrix(jnp.array([.2, -.1]))**2))(p))(.2) for v in (loose, tight))
    np.testing.assert_allclose(a, b, atol=2e-12)
    assert widths and max(widths) < provider.block_size


def test_mixed_precision_follows_actual_grid_operands_in_value_and_response():
    module = api()
    basis = module.Potential(grid(), jnp.eye(2, dtype=jnp.float32), functions, 2,
                             params=jnp.asarray(.2, dtype=jnp.float32))
    coefficients = jnp.array([.2, -.1], dtype=jnp.float32)
    actual = jax.jit(basis.matrix)(coefficients)
    assert actual.dtype == jnp.float64
    oracle = replace(basis, overlap=basis.overlap.astype(jnp.float64),
                     params=basis.params.astype(jnp.float64)).matrix(coefficients.astype(jnp.float64))
    np.testing.assert_allclose(actual, oracle, atol=2e-12)
    gradient = jax.jit(jax.grad(lambda c: jnp.sum(basis.matrix(c)**2)))(coefficients)
    assert gradient.dtype == jnp.float32 and np.isfinite(gradient).all()
