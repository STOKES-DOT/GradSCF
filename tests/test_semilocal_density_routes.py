"""Orbital hints may accelerate values but must preserve arbitrary density AD.

All inputs are small float64 arrays. Complex/nonsymmetric cases preserve the
existing bilinear (no conjugation) convention, not physical complex-MO SCF.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.tools import features


def _inputs():
    rng = np.random.default_rng(20261109)
    ao = jnp.asarray(rng.normal(scale=0.2, size=(7, 12)))
    derivative = jnp.asarray(rng.normal(scale=0.2, size=(4, 7, 12)))
    coeff = jnp.asarray(rng.normal(scale=0.3, size=(12, 9)))
    occupation = jnp.asarray([1.0, 0.35, 0., 0., 0., 0., 0., 0., 0.])
    density = _density(coeff, occupation)
    return ao, derivative, density, coeff, occupation


def _density(coeff, occupation):
    return jnp.einsum('pi,i,qi->pq', coeff, occupation, coeff,
                      precision=jax.lax.Precision.HIGHEST)


def _reference(ao, derivative, density, *, xc_kind):
    rho = jnp.einsum('rp,pq,rq->r', ao, density, ao)
    gradient = jnp.zeros((ao.shape[0], 3), dtype=rho.dtype)
    tau = jnp.zeros_like(rho)
    if xc_kind in ('GGA', 'MGGA'):
        gradient = 2 * jnp.einsum('xrp,pq,rq->rx', derivative[1:4], density, ao)
    if xc_kind == 'MGGA':
        tau = .5 * jnp.einsum('xrp,pq,xrq->r', derivative[1:4], density, derivative[1:4])
    return rho, gradient, tau


def _candidate(ao, derivative, density, coeff, occupation, *, xc_kind='MGGA', occupied_columns=2):
    # Resolve at call time so pre-implementation RED is an ordinary test failure,
    # rather than a collection failure hiding all independent AD contracts.
    assert hasattr(features, '_spin_density_fields'), 'missing density/orbital dispatch helper'
    return features._spin_density_fields(ao, derivative, density, xc_kind=xc_kind,
        mo_coeff=coeff, mo_occ=occupation, occupied_columns=occupied_columns)


def _close(actual, expected, tolerance=2e-12):
    actual = jax.tree_util.tree_leaves(actual)
    expected = jax.tree_util.tree_leaves(expected)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=tolerance, rtol=tolerance)


def _directions(operands):
    rng = np.random.default_rng(532)
    result = []
    for operand in operands:
        direction = rng.normal(scale=.07, size=operand.shape)
        if jnp.iscomplexobj(operand):
            direction = direction + 1j * rng.normal(scale=.04, size=operand.shape)
        result.append(jnp.asarray(direction, dtype=operand.dtype))
    return tuple(result)


def _check_response(function, reference, operands):
    directions = _directions(operands)
    function = jax.jit(function)
    _close(jax.jvp(function, operands, directions),
           jax.jvp(reference, operands, directions), 3e-11)
    output, pullback = jax.vjp(function, *operands)
    _, reference_pullback = jax.vjp(reference, *operands)
    cotangent = _directions(output)
    _close(pullback(cotangent), reference_pullback(cotangent), 3e-11)

    def loss(fn, args):
        values = fn(*args)
        return sum(jnp.real(jnp.vdot(value, value)) for value in values)

    gradient = jax.grad(lambda args: loss(function, args))
    reference_gradient = jax.grad(lambda args: loss(reference, args))
    _close(gradient(operands), reference_gradient(operands), 3e-11)
    _close(jax.jvp(gradient, (operands,), (directions,)),
           jax.jvp(reference_gradient, (operands,), (directions,)), 3e-10)

    def hvp(fn, args):
        return jax.jvp(fn, (args,), (directions,))[1]

    _close(jax.jvp(partial(hvp, gradient), (operands,), (directions,))[1],
           jax.jvp(partial(hvp, reference_gradient), (operands,), (directions,))[1], 3e-9)


@pytest.mark.parametrize('xc_kind', ['LDA', 'GGA', 'MGGA'])
def test_matching_orbitals_preserve_values_and_independent_density_response(xc_kind):
    ao, derivative, density, coeff, occupation = _inputs()
    candidate = partial(_candidate, xc_kind=xc_kind)
    reference = lambda a, da, d, c, o: _reference(a, da, d, xc_kind=xc_kind)
    _check_response(candidate, reference, (ao, derivative, density, coeff, occupation))


@pytest.mark.parametrize('xc_kind', ['LDA', 'GGA', 'MGGA'])
def test_physical_orbital_and_occupation_chain_rule_counted_once(xc_kind):
    ao, derivative, _, coeff, occupation = _inputs()
    def candidate(a, da, c, o):
        return _candidate(a, da, _density(c, o), c, o, xc_kind=xc_kind)
    def reference(a, da, c, o):
        return _reference(a, da, _density(c, o), xc_kind=xc_kind)
    _check_response(candidate, reference, (ao, derivative, coeff, occupation))


@pytest.mark.parametrize('case', ['mixed_density', 'tiny_mismatch', 'occupied_tail',
                                  'complex', 'nonsymmetric', 'nonfinite_hint'])
def test_unusable_orbital_hints_use_current_density(case):
    ao, derivative, density, coeff, occupation = _inputs()
    if case == 'mixed_density':
        density = .7 * density + .3 * jnp.eye(density.shape[0])
    elif case == 'tiny_mismatch':
        density = density.at[0, 0].add(1e-8)
    elif case == 'occupied_tail':
        occupation = occupation.at[4].set(.15)
        density = _density(coeff, occupation)
    elif case == 'complex':
        ao = ao + .12j * ao[::-1]
        derivative = derivative + .1j * derivative[:, ::-1]
        coeff = coeff + .2j * coeff[::-1]
        density = _density(coeff, occupation)
    elif case == 'nonsymmetric':
        density = density.at[1, 2].add(.19)
    else:
        coeff = coeff.at[0, 0].set(jnp.nan)
    candidate = lambda a, da, d: _candidate(a, da, d, coeff, occupation)
    reference = partial(_reference, xc_kind='MGGA')
    _check_response(candidate, reference, (ao, derivative, density))


def test_hint_is_revalidated_for_each_density_in_one_compiled_call():
    ao, derivative, density, coeff, occupation = _inputs()
    candidate = jax.jit(lambda d: _candidate(ao, derivative, d, coeff, occupation))
    for current in (density, .8 * density + .2 * jnp.eye(12), density):
        _close(candidate(current), _reference(ao, derivative, current, xc_kind='MGGA'))


@pytest.mark.parametrize('occupied_columns', [None, 0, 3, 9])
def test_static_rank_hints_do_not_change_results(occupied_columns):
    inputs = _inputs()
    _close(_candidate(*inputs, occupied_columns=occupied_columns),
           _reference(*inputs[:3], xc_kind='MGGA'))


def test_no_orbital_hint_and_lda_without_derivatives():
    ao, derivative, density, _, _ = _inputs()
    assert hasattr(features, '_spin_density_fields'), 'missing density/orbital dispatch helper'
    for kind in ('LDA', 'GGA', 'MGGA'):
        result = features._spin_density_fields(ao, None if kind == 'LDA' else derivative,
                                                density, xc_kind=kind)
        _close(result, _reference(ao, derivative, density, xc_kind=kind))
