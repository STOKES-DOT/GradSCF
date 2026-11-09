"""Measured MGGA orbital dispatch must preserve the physical XC operator.

Small synthetic float64 arrays exercise real r2SCAN energy and AO potentials.
The geometry check is an AO-grid operator response, not a converged molecular
nuclear gradient. No PySCF or full SCF calculation is used here.
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids.blocks import AOGrid
from gradscf.scf import rks, uks
from gradscf.tools import features


def _inputs(spin, nao=128, ngrids=256):
    rng = np.random.default_rng(9486)
    ao = jnp.asarray(rng.normal(scale=.1, size=(ngrids, nao)))
    derivative = jnp.asarray(rng.normal(scale=.1, size=(4, ngrids, nao))).at[0].set(ao)
    coeff = jnp.asarray(rng.normal(size=(2, nao, nao))/np.sqrt(nao))
    occ = jnp.zeros((2, nao)).at[0, :8].set(1.).at[1, :6].set(1.)
    if spin == 'restricted':
        coeff, occ = coeff[0], 2*occ[0]
    density = _density(coeff, occ)
    return ao, derivative, density, coeff, occ, jnp.ones(ngrids)/ngrids


def _density(coeff, occ):
    if coeff.ndim == 3:
        return jnp.stack([_density(c, o) for c, o in zip(coeff, occ)])
    return jnp.einsum('pi,i,qi->pq', coeff, occ, coeff,
                      precision=jax.lax.Precision.HIGHEST)


def _require_api(spin):
    function = (rks._restricted_xc_energy_and_matrix if spin == 'restricted'
                else uks._unrestricted_xc_energy_and_matrices)
    required = ({'mo_coeff', 'mo_occ', 'occupied_columns'} if spin == 'restricted' else
                {'mo_coeff_a', 'mo_coeff_b', 'mo_occ_a', 'mo_occ_b', 'nocc_a', 'nocc_b'})
    assert required <= inspect.signature(function).parameters.keys(), (
        f'{spin} XC consumer must accept redundant orbital hints and static occupied counts')
    return function


def _call(spin, values, *, kind='MGGA', hinted=True):
    function = _require_api(spin) if hinted else (
        rks._restricted_xc_energy_and_matrix if spin == 'restricted'
        else uks._unrestricted_xc_energy_and_matrices)
    ao, derivative, density, coeff, occ, weights = values
    options = dict(ao=ao, ao_deriv1=derivative, weights=weights, xc_kind=kind)
    name = 'r2scan' if kind == 'MGGA' else 'pbe'
    if spin == 'restricted':
        options.update(density=density, cfg=rks.RKSConfig(xc_spec=name))
        if hinted:
            options.update(mo_coeff=coeff, mo_occ=occ, occupied_columns=8)
    else:
        options.update(density_a=density[0], density_b=density[1], cfg=uks.UKSConfig(xc_spec=name))
        if hinted:
            options.update(mo_coeff_a=coeff[0], mo_coeff_b=coeff[1],
                           mo_occ_a=occ[0], mo_occ_b=occ[1], nocc_a=8, nocc_b=6)
    return function(**options)


def _close(left, right, tolerance=2e-9):
    left, right = jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
    assert len(left) == len(right)
    for actual, reference in zip(left, right):
        np.testing.assert_allclose(actual, reference, atol=tolerance, rtol=tolerance)


def _observe_fields(monkeypatch):
    original = features._spin_density_fields
    records = []
    def observed(ao, derivative, density, **kwargs):
        result = original(ao, derivative, density, **kwargs)
        records.append((ao, derivative, density, kwargs, result))
        return result
    monkeypatch.setattr(features, '_spin_density_fields', observed)
    # Consumers may import the common function or look it up on features.
    for module in (rks, uks):
        if hasattr(module, '_spin_density_fields'):
            monkeypatch.setattr(module, '_spin_density_fields', observed)
    return records


def _active(records):
    return [record for record in records if record[3].get('mo_coeff') is not None
            and record[3].get('occupied_columns') is not None]


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_eligible_mgga_consumer_reaches_common_orbital_fields(monkeypatch, spin):
    values = _inputs(spin)
    records = _observe_fields(monkeypatch)
    actual = _call(spin, values)
    active = list(_active(records))
    assert len(active) == (1 if spin == 'restricted' else 2)
    for ao, derivative, density, hints, result in active:
        assert hints['xc_kind'] == 'MGGA'
        assert hints['occupied_columns'] in (8, 6)
        np.testing.assert_array_equal(density, _density(hints['mo_coeff'], hints['mo_occ']))
        rho, grad = features._spin_density_and_gradient(ao, derivative, density)
        _close(result, (rho, grad, features._spin_tau(derivative, density)), 2e-12)
    _close(actual, _call(spin, values, hinted=False))


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
@pytest.mark.parametrize('kind,nao,ngrids', [('GGA', 128, 256), ('MGGA', 32, 256), ('MGGA', 128, 64)])
def test_unmeasured_or_small_cases_do_not_receive_automatic_hints(monkeypatch, spin, kind, nao, ngrids):
    values = _inputs(spin, nao, ngrids)
    records = _observe_fields(monkeypatch)
    actual = _call(spin, values, kind=kind)
    assert not _active(records), 'Automatic hints require measured MGGA AO/grid dimensions'
    _close(actual, _call(spin, values, kind=kind, hinted=False))


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
@pytest.mark.parametrize('case', ['mixed_density', 'stale_coefficients', 'occupied_tail'])
def test_initial_or_mixed_density_and_tail_occupations_keep_current_density(spin, case):
    ao, derivative, density, coeff, occ, weights = _inputs(spin)
    if case == 'mixed_density':
        density = .8*density + .02*jnp.eye(density.shape[-1])
    elif case == 'stale_coefficients':
        coeff = 1.03*coeff
    else:
        occ = occ.at[..., 17].set(.05)
        density = _density(coeff, occ)
    values = (ao, derivative, density, coeff, occ, weights)
    _close(_call(spin, values), _call(spin, values, hinted=False))


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_compiled_xc_hints_preserve_arbitrary_density_response_and_revalidation(spin):
    ao, derivative, density, coeff, occ, weights = _inputs(spin)
    rng = np.random.default_rng(619)
    direction = jnp.asarray(rng.normal(scale=.005, size=density.shape))
    candidate = jax.jit(lambda d: _call(spin, (ao, derivative, d, coeff, occ, weights)))
    dense = lambda d: _call(spin, (ao, derivative, d, coeff, occ, weights), hinted=False)
    _close(jax.jvp(candidate, (density,), (direction,)),
           jax.jvp(dense, (density,), (direction,)))
    output, pullback = jax.vjp(candidate, density)
    _, dense_pullback = jax.vjp(dense, density)
    cotangent = (jnp.asarray(.7), jnp.asarray(rng.normal(scale=.01, size=output[1].shape)))
    _close(pullback(cotangent), dense_pullback(cotangent))
    gradient = jax.grad(lambda d: candidate(d)[0])
    dense_gradient = jax.grad(lambda d: dense(d)[0])
    _close(jax.jvp(gradient, (density,), (direction,)),
           jax.jvp(dense_gradient, (density,), (direction,)))
    mixed = .85*density + .03*jnp.eye(density.shape[-1])
    for current in (mixed, density):
        _close(candidate(current), dense(current))


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_ao_center_and_exponent_response_reaches_physical_grid_inputs(spin):
    _, _, density, coeff, occ, weights = _inputs(spin)
    rng = np.random.default_rng(920)
    centers = jnp.asarray(rng.normal(scale=.4, size=(128, 3)))
    coords = jnp.asarray(rng.normal(scale=.5, size=(256, 3)))
    exponents = jnp.asarray(rng.uniform(.6, 1.3, size=(128, 1)))
    def energy(parameter, hinted):
        provider = AOGrid(coords, weights, centers+parameter*jnp.asarray([.2, -.1, .3]),
            exponents*jnp.exp(.1*parameter), jnp.full((128, 1), .2),
            (1,)*128, ((0, 0, 0),)*128, 256)
        ao, derivative, _, _, grid_weights = provider.block(0, 256, deriv=1)
        return _call(spin, (ao, derivative, density, coeff, occ, grid_weights), hinted=hinted)[0]
    candidate = jax.jit(lambda p: energy(p, True))
    dense = lambda p: energy(p, False)
    point, direction = jnp.asarray(.03), jnp.asarray(.2)
    _close(jax.value_and_grad(candidate)(point), jax.value_and_grad(dense)(point))
    _close(jax.jvp(jax.grad(candidate), (point,), (direction,)),
           jax.jvp(jax.grad(dense), (point,), (direction,)))
    for step in (1e-4, 5e-5):
        finite = (candidate(point+step)-candidate(point-step))/(2*step)
        _close(finite, jax.grad(candidate)(point), 2e-7)


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_fock_builders_forward_dynamic_orbitals_and_static_counts(monkeypatch, spin):
    ao, derivative, density, coeff, occ, weights = values = _inputs(spin)
    _require_api(spin)
    records = _observe_fields(monkeypatch)
    zero = jnp.zeros((128, 128))
    if spin == 'restricted':
        assert 'occupied_columns' in inspect.signature(rks._fock_components_for_density).parameters
        _, _, xc, fock = rks._fock_components_for_density(density=density, mo_coeff=coeff,
            mo_occ=occ, mo_energy=jnp.zeros(128), occupied_columns=8,
            jk_builder=lambda *_: (zero, zero), ao=ao, ao_deriv1=derivative,
            weights=weights, h=zero, alpha=jnp.asarray(0.),
            cfg=rks.RKSConfig(xc_spec='r2scan'), xc_kind='MGGA')
        actual = xc.energy, fock
    else:
        _, energy, fa, fb = uks._raw_fock_and_energy_for_state(density_a=density[0],
            density_b=density[1], ao=ao, ao_deriv1=derivative, weights=weights,
            h=zero, eri=jnp.zeros((0,)), df_factors=None, enuc=jnp.asarray(0.),
            alpha=jnp.asarray(0.), cfg=uks.UKSConfig(xc_spec='r2scan'), xc_kind='MGGA',
            mo_coeff_a=coeff[0], mo_coeff_b=coeff[1], mo_occ_a=occ[0], mo_occ_b=occ[1],
            nocc_a=8, nocc_b=6, jk_matrices=(zero, zero, zero))
        actual = energy, jnp.stack([fa, fb])
    assert len(_active(records)) == (1 if spin == 'restricted' else 2)
    _close(actual, _call(spin, values, hinted=False))
