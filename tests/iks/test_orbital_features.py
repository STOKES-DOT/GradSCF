"""Frozen orbital preparation and differentiable invariant grid descriptors."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _grid(nao=4, *, block_size=3, ngrids=7):
    from gradscf.integrals.grids.blocks import AOGrid

    return AOGrid(jnp.asarray([[.11*i, .2, -.07*i] for i in range(ngrids)]),
        jnp.linspace(.1, .3, ngrids),
        jnp.asarray([[.23*i, .04*i, -.1*i] for i in range(nao)]),
        jnp.linspace(.7, 1.3, nao)[:, None], jnp.ones((nao, 1)),
        (1,)*nao, ((0, 0, 0),)*nao, block_size)


def _reference(layout='rks', *, energies=None):
    from gradscf.dft.iks import OrbitalReference, Target

    overlap = jnp.diag(jnp.asarray([1.2, .8, 1.1, .9]))
    angle = .31
    frame = jnp.asarray([[jnp.cos(angle), 0., -jnp.sin(angle), 0.],
                         [0., 1., 0., 0.],
                         [jnp.sin(angle), 0., jnp.cos(angle), 0.],
                         [0., 0., 0., 1.]])
    coefficients = jnp.diag(1/jnp.sqrt(jnp.diag(overlap)))@frame
    occupations = jnp.asarray([2., 2., 0., 0.])
    energies = jnp.asarray([-.8, -.8, .3, .3] if energies is None else energies)
    density = jnp.diag(jnp.asarray([1.7, 1.3, .6, .4])/jnp.diag(overlap))
    nelec = 4
    if layout == 'uks':
        coefficients = jnp.stack((coefficients, coefficients))
        occupations = jnp.stack((occupations/2, occupations/2))
        energies = jnp.stack((energies, energies))
        density = jnp.stack((density/2, density/2))
        nelec = (2, 2)
    target = Target(density, overlap, nelec, layout=layout)
    return OrbitalReference(target, coefficients, occupations, energies,
                            method='declared-reference')


def _dense_oracle(reference, grid, windows, transform=None):
    # Independent orbital-value contraction; production must not cache this.
    ao, derivative, _, _, _ = grid.block(0, grid.ngrids, deriv=1)
    coefficients = np.asarray(reference.coefficients)
    occupations = np.asarray(reference.occupations)
    energies = np.asarray(reference.energies)
    if reference.target.layout == 'rks':
        coefficients, occupations, energies = (a[None] for a in
                                                (coefficients, occupations, energies))
    maximum = 2 if reference.target.layout == 'rks' else 1
    result = {name: np.zeros(grid.ngrids) for name in
              ('rho_occ', 'tau_occ', *(f'virtual_{i}' for i in range(len(windows))))}
    for c, occ, eps in zip(coefficients, occupations, energies):
        cartesian = c if transform is None else np.asarray(transform)@c
        phi = np.asarray(ao)@cartesian
        gradient = np.einsum('xgp,pi->xgi', np.asarray(derivative)[1:4], cartesian)
        result['rho_occ'] += np.sum(phi**2*occ, axis=1)
        result['tau_occ'] += .5*np.einsum('xgi,i,xgi->g', gradient, occ, gradient)
        occupied = occ > 0
        mu = .5*(np.max(eps[occupied])+np.min(eps[~occupied]))
        for index, eta in enumerate(windows):
            weight = maximum*(~occupied)*eta**2/((eps-mu)**2+eta**2)
            result[f'virtual_{index}'] += np.sum(phi**2*weight, axis=1)
    return result


@pytest.mark.parametrize('layout', ['rks', 'uks'])
@pytest.mark.parametrize('use_transform', [False, True])
def test_orbital_grid_fields_match_independent_matrix_contraction(layout, use_transform):
    from gradscf.dft.iks.orbital_features import orbital_grid_features

    reference = _reference(layout, energies=[-.9, -.6, .2, .7])
    grid = _grid(5 if use_transform else 4)
    transform = jnp.asarray([[1., 0., .1, 0.], [0., 1., 0., .1],
                             [0., 0., .8, 0.], [.1, 0., 0., 1.],
                             [.2, -.1, .1, .3]]) if use_transform else None
    windows = (.2, 1.)
    actual = orbital_grid_features(reference, grid, transform=transform,
                                    windows=windows, max_memory=1.)
    expected = _dense_oracle(reference, grid, windows, transform)
    assert set(actual) == {'rho_occ', 'tau_occ', 'virtual_0', 'virtual_1'}
    for name, value in actual.items():
        assert value.shape == (grid.ngrids,) and not jnp.iscomplexobj(value)
        np.testing.assert_allclose(value, expected[name], atol=2e-12, rtol=2e-12)
    assert reference.target.energy is None
    assert np.linalg.norm(reference.density-reference.target.density) > .01


@pytest.mark.parametrize('layout', ['rks', 'uks'])
def test_orbital_fields_ignore_phase_permutation_and_degenerate_rotations(layout):
    from gradscf.dft.iks.orbital_features import orbital_grid_features

    reference, grid = _reference(layout), _grid()
    expected = orbital_grid_features(reference, grid, max_memory=1.)
    angle = .47
    rotation = jnp.asarray([[jnp.cos(angle), -jnp.sin(angle)],
                            [jnp.sin(angle), jnp.cos(angle)]])
    frame = jnp.zeros((4, 4)).at[:2, :2].set(rotation).at[2:, 2:].set(rotation.T)
    permutation = jnp.asarray([3, 0, 2, 1])
    coefficients = (reference.coefficients@frame)[..., permutation]
    coefficients = coefficients*jnp.asarray([-1., 1., -1., 1.])
    changed = replace(reference, coefficients=coefficients,
                      occupations=reference.occupations[..., permutation],
                      energies=reference.energies[..., permutation])
    actual = orbital_grid_features(changed, grid, max_memory=1.)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-12, rtol=2e-12)


def test_equivalent_restricted_and_unrestricted_fields_have_equal_spin_sum():
    from gradscf.dft.iks.orbital_features import orbital_grid_features

    rks = orbital_grid_features(_reference(), _grid(), max_memory=1.)
    uks = orbital_grid_features(_reference('uks'), _grid(), max_memory=1.)
    for name in rks:
        np.testing.assert_allclose(rks[name], uks[name], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('windows', [(0.,), (-.2,), (np.inf,), (np.nan,), (.2j,), (True,)])
def test_virtual_energy_windows_require_positive_finite_hartree_widths(windows):
    from gradscf.dft.iks.orbital_features import orbital_grid_features

    with pytest.raises(ValueError, match='window|width|Hartree'):
        orbital_grid_features(_reference(), _grid(), windows=windows)


def test_fractional_or_missing_occupied_virtual_split_is_rejected():
    from gradscf.dft.iks import OrbitalReference, Target
    from gradscf.dft.iks.orbital_features import orbital_grid_features

    reference = _reference()
    fractional = replace(reference, occupations=jnp.asarray([1.5, 1.5, .5, .5]))
    with pytest.raises(ValueError, match='integer'):
        orbital_grid_features(fractional, _grid())
    for count in (0, 4):
        occupations = jnp.full((4,), 2. if count else 0.)
        density = (reference.coefficients*occupations)@reference.coefficients.T
        target = Target(density, reference.overlap, 2*count)
        complete = OrbitalReference(target, reference.coefficients, occupations,
                                     reference.energies, method='split-test')
        with pytest.raises(ValueError, match='occupied|virtual|split'):
            orbital_grid_features(complete, _grid())
    unrestricted = _reference('uks')
    occupations = unrestricted.occupations.at[1].set(0.)
    target = replace(unrestricted.target,
        density=unrestricted.target.density.at[1].set(0.), nelec=(2, 0))
    empty_spin = replace(unrestricted, target=target, occupations=occupations)
    with pytest.raises(ValueError, match='occupied|virtual|split'):
        orbital_grid_features(empty_spin, _grid())


def test_grid_memory_budget_charges_retained_outputs_before_ao_evaluation(monkeypatch):
    from gradscf.dft.iks.orbital_features import orbital_grid_features
    from gradscf.integrals.grids.blocks import AOGrid

    grid = _grid(ngrids=257)
    reference = _reference()
    def forbidden(*args, **kwargs):
        pytest.fail('AO evaluated before rejecting an impossible retained-output budget')
    monkeypatch.setattr(AOGrid, 'block', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident|quadrature'):
        orbital_grid_features(reference, grid, max_memory=.02)


@pytest.mark.parametrize('parameter', ['coefficients', 'energies', 'geometry'])
def test_supplied_orbital_feature_derivatives_match_finite_difference(parameter):
    from gradscf.dft.iks.orbital_features import orbital_grid_features

    reference = _reference(energies=[-.9, -.6, .2, .7])
    provider = _grid()
    def objective(t):
        updated, grid = reference, provider
        if parameter == 'coefficients':
            rotation = jnp.eye(4).at[0, 0].set(jnp.cos(t)).at[2, 2].set(jnp.cos(t))
            rotation = rotation.at[0, 2].set(-jnp.sin(t)).at[2, 0].set(jnp.sin(t))
            updated = replace(reference, coefficients=reference.coefficients@rotation)
        elif parameter == 'energies':
            updated = replace(reference, energies=reference.energies+t*jnp.asarray([.1, -.2, .3, .4]))
        else:
            grid = replace(provider, centers=provider.centers+t*.07,
                            coords=provider.coords+t*.03)
        fields = orbital_grid_features(updated, grid, max_memory=1.)
        return jnp.sum(provider.weights*(fields['rho_occ']+.3*fields['tau_occ']
                                          +.7*fields['virtual_0']+.2*fields['virtual_1']))
    point, step = .12, 1e-5
    derivative = jax.jit(jax.grad(objective))(jnp.asarray(point))
    finite_difference = (objective(point+step)-objective(point-step))/(2*step)
    assert np.isfinite(derivative) and abs(float(derivative)) > 1e-6
    np.testing.assert_allclose(derivative, finite_difference, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize('layout', ['rks', 'uks'])
def test_prepare_reference_keeps_complete_spectrum_at_internal_degeneracy(layout):
    from gradscf.dft.iks.orbital_features import prepare_orbital_reference

    source = _reference(layout)
    actual = prepare_orbital_reference(source.target, source.fock,
                                       method='PBE@fixed-target-density', max_memory=1.)
    assert actual.coefficients.shape == source.coefficients.shape
    assert actual.method == 'PBE@fixed-target-density' and bool(actual.valid)
    np.testing.assert_allclose(actual.energies, source.energies, atol=2e-12)
    np.testing.assert_allclose(actual.occupations, source.occupations, atol=2e-12)
    np.testing.assert_allclose(actual.density, source.density, atol=2e-12)
    np.testing.assert_allclose(actual.fock, source.fock, atol=2e-12)


def test_prepare_reference_rejects_traced_fock_instead_of_exposing_zero_ad():
    from gradscf.dft.iks.orbital_features import prepare_orbital_reference

    source = _reference()
    def objective(fock):
        reference = prepare_orbital_reference(source.target, fock, method='offline')
        return jnp.sum(reference.coefficients)
    with pytest.raises(NotImplementedError, match='offline|frozen|prepar'):
        jax.jit(objective)(source.fock)
    with pytest.raises(NotImplementedError, match='offline|frozen|prepar'):
        jax.grad(objective)(source.fock)


def test_prepare_reference_checks_real_finite_symmetric_fock_and_memory():
    from gradscf.dft.iks.orbital_features import prepare_orbital_reference

    source = _reference()
    for fock in (source.fock.astype(complex), source.fock.at[0, 1].add(.1),
                 source.fock.at[0, 0].set(jnp.nan)):
        with pytest.raises(ValueError, match='real|finite|symmetric|Fock|fock'):
            prepare_orbital_reference(source.target, fock, method='offline')
    with pytest.raises(ValueError, match='memory|budget|resident'):
        prepare_orbital_reference(source.target, source.fock, method='offline', max_memory=1e-6)


def test_orbital_grid_map_retiles_workspace_without_full_grid_orbital_cache(monkeypatch):
    from gradscf.dft.iks.orbital_features import orbital_grid_features
    from gradscf.integrals.grids.blocks import AOGrid

    reference, provider = _reference(), _grid(block_size=128)
    expected = orbital_grid_features(reference, provider, max_memory=1.)
    widths = []
    original = AOGrid.block
    def observed(self, start, size, *, deriv=0):
        widths.append(size)
        return original(self, start, size, deriv=deriv)
    monkeypatch.setattr(AOGrid, 'block', observed)
    actual = orbital_grid_features(reference, provider, max_memory=.02)
    assert widths and max(widths) < provider.block_size
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-12, rtol=2e-12)


def test_empty_windows_only_require_integer_occupied_density():
    from gradscf.dft.iks import OrbitalReference, Target
    from gradscf.dft.iks.orbital_features import orbital_grid_features

    source = _reference()
    occupations = jnp.full((4,), 2.)
    density = (source.coefficients*occupations)@source.coefficients.T
    target = Target(density, source.overlap, 8)
    reference = OrbitalReference(target, source.coefficients, occupations, source.energies)
    fields = orbital_grid_features(reference, _grid(), windows=(), max_memory=1.)
    assert set(fields) == {'rho_occ', 'tau_occ'}
    assert all(np.isfinite(value).all() for value in fields.values())


@pytest.mark.parametrize('layout', ['rks', 'uks'])
def test_prepare_memory_budget_precedes_target_spectral_validation(monkeypatch, layout):
    from gradscf.dft.iks.orbital_features import prepare_orbital_reference

    source = _reference(layout)
    fock = source.fock
    def forbidden(*args, **kwargs):
        raise AssertionError('Spectral target validation preceded the memory guard')
    monkeypatch.setattr(jnp.linalg, 'eigvalsh', forbidden)
    monkeypatch.setattr(jnp.linalg, 'cholesky', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        prepare_orbital_reference(source.target, fock, method='offline', max_memory=1e-6)


@pytest.mark.parametrize('layout', ['rks', 'uks'])
def test_grid_memory_budget_precedes_reference_spectral_validation(monkeypatch, layout):
    from gradscf.dft.iks.orbital_features import orbital_grid_features

    reference, provider = _reference(layout), _grid()
    def forbidden(*args, **kwargs):
        raise AssertionError('Spectral reference validation preceded the memory guard')
    monkeypatch.setattr(jnp.linalg, 'eigvalsh', forbidden)
    monkeypatch.setattr(jnp.linalg, 'cholesky', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        orbital_grid_features(reference, provider, max_memory=1e-6)
