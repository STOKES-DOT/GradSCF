"""Semilocal array contractions preserve the old algebra while sharing work.

These tiny array tests do not establish complex-orbital SCF support. In
particular, the existing transpose (not conjugate-transpose) convention and
nonsymmetric-density primitive are retained exactly.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import gks, rks, uks
from gradscf.tools.features import _spin_density_and_gradient

jax.config.update('jax_enable_x64', True)


def arrays(complex_values=False):
    rng = np.random.default_rng(20261105)
    def sample(shape):
        value = .2 * rng.normal(size=shape)
        if complex_values: value = value + .15j * rng.normal(size=shape)
        return jnp.asarray(value)
    return (sample((7, 4)), sample((4, 7, 4)), sample((7, 4)),
            jnp.asarray(rng.uniform(.03, .09, 7)), sample((7,)), sample((7, 3)),
            sample((7,)), sample((7,)))


def density_oracle(ao, derivative, density):
    # Independent previous formulas; do not symmetrize density or add conjugates.
    rho = jnp.einsum('rp,pq,rq->r', ao, density, ao)
    grad = 2 * jnp.einsum('xrp,pq,rq->rx', derivative[1:4], density, ao)
    return rho, grad


def potential_oracle(ao, derivative, laplacian, weights, vrho, vgrad, vtau, vlapl, *, kind):
    matrix = jnp.einsum('r,rp,rq->pq', weights * vrho, ao, ao)
    if kind in ('GGA', 'MGGA', 'MGGA_LAPL'):
        term = jnp.einsum('rx,xrp,rq->pq', weights[:, None] * vgrad, derivative[1:4], ao)
        matrix = matrix + term + term.T
    if kind in ('MGGA', 'MGGA_LAPL'):
        matrix = matrix + .5 * jnp.einsum('r,xrp,xrq->pq', weights * vtau, derivative[1:4], derivative[1:4])
    if kind == 'MGGA_LAPL':
        left = jnp.einsum('r,rp,rq->pq', weights * vlapl, laplacian, ao)
        gradient = 2 * jnp.einsum('r,xrp,xrq->pq', weights * vlapl, derivative[1:4], derivative[1:4])
        matrix = matrix + left + left.T + gradient
    return .5 * (matrix + matrix.T)


def potential(ao, derivative, laplacian, weights, vrho, vgrad, vtau, vlapl, *, kind):
    return rks._vxc_matrix_from_grid_potential(ao=ao, ao_deriv1=derivative,
        ao_laplacian=laplacian, weights=weights, vxc_rho=vrho, vxc_grad=vgrad,
        vxc_tau=vtau, vxc_lapl=vlapl, xc_kind=kind)


def assert_tree_close(actual, expected, *, tolerance):
    got, reference = jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    assert len(got) == len(reference)
    for a, b in zip(got, reference):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


def check_array_derivatives(function, oracle, operands):
    """Joint AO/dAO/density or quadrature/potential directions, including HVP."""
    rng = np.random.default_rng(581)
    def direction(value):
        result = .07 * rng.normal(size=value.shape)
        if jnp.iscomplexobj(value): result = result + .05j * rng.normal(size=value.shape)
        return jnp.asarray(result, dtype=value.dtype)
    tangent = tuple(direction(value) for value in operands)
    compiled = jax.jit(function)
    actual, derivative = jax.jvp(compiled, operands, tangent)
    expected, oracle_derivative = jax.jvp(oracle, operands, tangent)
    assert_tree_close(actual, expected, tolerance=2e-12)
    assert_tree_close(derivative, oracle_derivative, tolerance=2e-11)
    cotangent = jax.tree_util.tree_map(direction, actual)
    _, pullback = jax.vjp(compiled, *operands)
    _, reference_pullback = jax.vjp(oracle, *operands)
    assert_tree_close(pullback(cotangent), reference_pullback(cotangent), tolerance=2e-11)
    def loss(fn, values):
        return sum(jnp.real(jnp.vdot(c, y)) for c, y in zip(
            jax.tree_util.tree_leaves(cotangent), jax.tree_util.tree_leaves(fn(*values))))
    gradient = jax.grad(lambda values: loss(compiled, values))
    reference_gradient = jax.grad(lambda values: loss(oracle, values))
    hvp = jax.jvp(gradient, (operands,), (tangent,))[1]
    reference_hvp = jax.jvp(reference_gradient, (operands,), (tangent,))[1]
    assert_tree_close(hvp, reference_hvp, tolerance=2e-10)
    changed = tuple(value + .03 * delta for value, delta in zip(operands, tangent))
    assert_tree_close(compiled(*changed), oracle(*changed), tolerance=2e-12)


@pytest.mark.parametrize('complex_values', [False, True])
def test_density_and_gradient_share_projection_without_changing_array_derivatives(complex_values):
    ao, derivative, *_ = arrays(complex_values)
    rng = np.random.default_rng(72)
    density = .3 * rng.normal(size=(4, 4))
    if complex_values: density = density + .2j * rng.normal(size=(4, 4))
    assert not np.allclose(density, density.T)
    check_array_derivatives(_spin_density_and_gradient, density_oracle,
                            (ao, derivative, jnp.asarray(density)))


@pytest.mark.parametrize('kind', ['LDA', 'GGA', 'MGGA', 'MGGA_LAPL'])
@pytest.mark.parametrize('complex_values', [False, True])
def test_potential_preserves_all_channels_and_array_derivatives(kind, complex_values):
    check_array_derivatives(partial(potential, kind=kind),
                            partial(potential_oracle, kind=kind), arrays(complex_values))


@pytest.mark.parametrize('kind', ['GGA', 'MGGA'])
def test_unrestricted_potential_keeps_independent_spin_and_tau_channels(kind):
    ao, derivative, laplacian, weights, vrho, vgrad, vtau, _ = arrays()
    operands = (ao, derivative, laplacian, weights, vrho, -.6 * vrho,
                vgrad, 1.3 * vgrad, vtau, -.4 * vtau)
    def production(a, da, lap, w, ra, rb, ga, gb, ta, tb):
        return uks._unrestricted_vxc_matrices(ao=a, ao_deriv1=da, ao_laplacian=lap,
            weights=w, vxc_rho_a=ra, vxc_rho_b=rb, vxc_grad_a=ga, vxc_grad_b=gb,
            vxc_tau_a=ta, vxc_tau_b=tb, xc_kind=kind)
    def reference(a, da, lap, w, ra, rb, ga, gb, ta, tb):
        return tuple(potential_oracle(a, da, lap, w, r, g, t, jnp.zeros_like(r), kind=kind)
                     for r, g, t in ((ra, ga, ta), (rb, gb, tb)))
    check_array_derivatives(production, reference, operands)


def test_uks_and_collinear_gks_reuse_the_existing_potential_helper():
    assert uks._vxc_matrix_from_grid_potential is rks._vxc_matrix_from_grid_potential
    assert gks._unrestricted_vxc_matrices is uks._unrestricted_vxc_matrices


def equations(graph):
    """Walk primitive jaxprs, including nested jit calls, without HLO text."""
    graph = getattr(graph, 'jaxpr', graph)
    if not hasattr(graph, 'eqns'): return
    for equation in graph.eqns:
        yield equation
        for value in equation.params.values():
            for nested in value if isinstance(value, (tuple, list)) else (value,):
                yield from equations(nested)


def test_density_projection_executes_once_in_the_array_graph():
    ao, derivative, *_ = arrays()
    density = jnp.arange(16, dtype=jnp.float64).reshape(4, 4) * .01
    graph = jax.make_jaxpr(_spin_density_and_gradient)(ao, derivative, density)
    projections = []
    for equation in equations(graph):
        shapes = [getattr(value.aval, 'shape', ()) for value in equation.invars]
        outputs = [getattr(value.aval, 'shape', ()) for value in equation.outvars]
        if (equation.primitive.name == 'dot_general' and (4, 4) in shapes and (7, 4) in shapes
                and any(shape in ((7, 4), (4, 7)) for shape in outputs)):
            projections.append(equation)
    assert len(projections) == 1, f'Expected one shared AO-density projection, found {len(projections)}'


def test_gga_potential_executes_one_gram_in_the_array_graph():
    graph = jax.make_jaxpr(partial(potential, kind='GGA'))(*arrays())
    grams = [equation for equation in equations(graph) if equation.primitive.name == 'dot_general'
             and any(getattr(value.aval, 'shape', ()) == (4, 4) for value in equation.outvars)]
    assert len(grams) == 1, f'Expected one weighted-AO Gram per spin, found {len(grams)}'
