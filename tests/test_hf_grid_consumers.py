"""T4.2: HF energy is grid-free; reference consumers lazily recover the grid.

Grid/integral execution is replaced by metadata fixtures in this narrow suite.
Existing neural tests supply the physical fixed-density/explicit/implicit gates.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gradscf.scf import builders
from gradscf.scf.inputs import assembly
from gradscf.scf.inputs.types import RKSIntegralInputs, UKSIntegralInputs
from gradscf.scf.rks import RKSConfig, RKSResult
from gradscf.scf.uks import UKSConfig


def _inputs(unrestricted=False):
    basis = SimpleNamespace(nao=2, atom_coords=np.zeros((1, 3)), atom_charges=np.array([2.]))
    common = dict(basis=basis, overlap=np.eye(2), hcore=-np.eye(2),
        eri=np.zeros((2, 2, 2, 2)), df_factors=np.ones((1, 2, 2)), direct_basis=object(),
        nuclear_repulsion=0., coords=np.zeros((0, 3)), grid_weights=np.zeros(0),
        ao=np.zeros((0, 2)), ao_deriv1=np.zeros((4, 0, 2)), ao_laplacian=None,
        dipole_integrals=np.zeros((3, 2, 2)))
    if unrestricted:
        return UKSIntegralInputs(**common, nalpha=1, nbeta=1, total_electrons=2)
    return RKSIntegralInputs(**common, nelectron=2, eri_pair_matrix=None,
        init_density=np.eye(2), init_mo_coeff=np.eye(2), init_mo_occ=np.array([2., 0.]),
        init_mo_energy=np.array([-1., 1.]))


def _grid_fixture(monkeypatch, inputs, streamed=False):
    weights = np.array([.2, .3, .4])
    provider = object() if streamed else None
    context = SimpleNamespace(basis=inputs.basis, coords=np.ones((3, 3)),
        grid_weights=weights, grid_ao=provider)
    ao = np.zeros((0 if streamed else 3, 2))
    derivative = np.zeros((4, len(ao), 2))
    calls = []

    def prepare(**kwargs):
        assert kwargs['needs_grid'] is True
        assert kwargs['grids_level'] == 3 and kwargs['max_l'] == 3
        calls.append(kwargs)
        return context

    def payload(current, **kwargs):
        assert current is context
        return ao, derivative, None

    monkeypatch.setattr(assembly, '_prepare_basis_grid_context', prepare)
    monkeypatch.setattr(assembly, '_grid_ao_payload', payload)
    return context, ao, derivative, calls


@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('streamed', [False, True])
def test_hf_grid_completion_is_lazy_and_preserves_spatial_inputs(monkeypatch, unrestricted, streamed):
    inputs = _inputs(unrestricted)
    context, ao, derivative, calls = _grid_fixture(monkeypatch, inputs, streamed)
    cfg = (UKSConfig if unrestricted else RKSConfig)(xc_spec='hf', grid_block_size=2 if streamed else -1)

    def forbidden(*args, **kwargs):
        raise AssertionError('Grid completion must not rebuild spatial integrals or rerun SCF')

    for name in ('prepare_basis', 'make_plan', 'build_rks_integral_inputs', 'build_uks_integral_inputs'):
        monkeypatch.setattr(assembly, name, forbidden)
    monkeypatch.setattr(builders, 'run_rks_from_integrals', forbidden)
    monkeypatch.setattr(builders, 'run_uks_from_integrals', forbidden)
    kwargs = dict(require_grid=False, config=cfg, grids_level=3, max_l=3)
    assert builders.complete_restricted_response_inputs(inputs, object(), 'fixture', **kwargs) is inputs
    assert calls == []
    kwargs['require_grid'] = True
    completed = builders.complete_restricted_response_inputs(inputs, object(), 'fixture', **kwargs)
    assert len(calls) == 1
    assert completed.coords is context.coords and completed.grid_weights is context.grid_weights
    assert completed.ao is ao and completed.ao_deriv1 is derivative
    assert completed.grid_ao is context.grid_ao
    for name in ('overlap', 'hcore', 'eri', 'df_factors', 'direct_basis', 'dipole_integrals'):
        assert getattr(completed, name) is getattr(inputs, name)
    if not unrestricted:
        for name in ('init_density', 'init_mo_coeff', 'init_mo_occ', 'init_mo_energy'):
            assert getattr(completed, name) is getattr(inputs, name)
    assert builders.complete_restricted_response_inputs(completed, object(), 'fixture', **kwargs) is completed
    assert len(calls) == 1


def test_restricted_hf_cached_reference_completes_grid_before_hfx(monkeypatch):
    inputs = _inputs()
    _, ao, _, calls = _grid_fixture(monkeypatch, inputs)
    result = RKSResult(converged=True, total_energy=-2., electronic_energy=-2.,
        nuclear_repulsion=0., xc_energy=0., exact_exchange_fraction=1., cycles=1,
        mo_coeff=inputs.init_mo_coeff, mo_occ=inputs.init_mo_occ,
        mo_energy=np.array([-1., -1.]), density_matrix=np.diag([2., 0.]),
        fock_matrix=inputs.hcore, overlap_matrix=inputs.overlap, hcore_matrix=inputs.hcore)

    class ReachedHFX(Exception):
        pass

    def hfx(basis, values, densities, coords, **kwargs):
        assert values is ao and len(coords) == 3
        raise ReachedHFX

    def forbidden(*args, **kwargs):
        raise AssertionError('Cached reference must not rerun SCF or integral assembly')

    monkeypatch.setattr(builders, '_local_hfx_features_from_basis_dm', hfx)
    monkeypatch.setattr(builders, 'build_rks_integral_inputs', forbidden)
    monkeypatch.setattr(builders, 'run_rks_from_integrals', forbidden)
    with pytest.raises(ReachedHFX):
        builders._build_restricted_reference(atom='He 0 0 0', basis='fixture', xc_spec='hf',
            grids_level=3, rks_config=RKSConfig(xc_spec='hf', grid_block_size=-1),
            scf_inputs=inputs, scf_result=result, compute_local_hfx_features=True)
    assert len(calls) == 1


def test_unrestricted_hf_reference_has_grid_for_neural_initializer(monkeypatch):
    inputs = _inputs(True)
    context, _, _, calls = _grid_fixture(monkeypatch, inputs)
    monkeypatch.setattr(builders, 'build_uks_integral_inputs', lambda **kwargs: inputs)

    class ReachedSCF(Exception):
        pass

    def run(**kwargs):
        assert kwargs['grid_weights'] is context.grid_weights
        assert kwargs['ao'].shape == (3, 2)
        raise ReachedSCF

    monkeypatch.setattr(builders, 'run_uks_from_integrals', run)
    with pytest.raises(ReachedSCF):
        builders._build_unrestricted_reference(atom='He 0 0 0', basis='fixture', xc_spec='hf',
            grids_level=3, uks_config=UKSConfig(xc_spec='hf', grid_block_size=-1))
    assert len(calls) == 1


def _uhf_facade_fixture(cart=True):
    from gradscf import gto, scf
    from gradscf.scf.uks import UKSResult

    inputs = _inputs(True)
    coeff_a, coeff_b = np.eye(2), np.eye(2)[:, ::-1].copy()
    occ = np.array([1., 0.])
    result = UKSResult(converged=True, total_energy=-2., electronic_energy=-2.,
        nuclear_repulsion=0., xc_energy=0., exact_exchange_fraction=1., cycles=2,
        mo_coeff_alpha=coeff_a, mo_coeff_beta=coeff_b,
        mo_occ_alpha=occ, mo_occ_beta=occ.copy(),
        mo_energy_alpha=np.array([-1., 1.]), mo_energy_beta=np.array([-.8, 1.2]),
        density_matrix_alpha=np.diag([1., 0.]), density_matrix_beta=np.diag([0., 1.]),
        fock_matrix_alpha=np.diag([-1., 1.]), fock_matrix_beta=np.diag([1.2, -.8]),
        overlap_matrix=inputs.overlap, hcore_matrix=inputs.hcore)
    mf = scf.UHF(gto.M(atom='He 0 0 0', basis='sto-3g', cart=cart), grids_level=3,
        grid_block_size=-1, execution_device='cpu')
    return mf, inputs, result


def test_ordinary_uhf_kernel_defers_grid_and_reference_build(monkeypatch):
    mf, inputs, result = _uhf_facade_fixture()
    _, _, _, grid_calls = _grid_fixture(monkeypatch, inputs)
    counts = {'integrals': 0, 'scf': 0}
    seen = []

    def build(**kwargs):
        counts['integrals'] += 1
        return inputs

    def run(**kwargs):
        counts['scf'] += 1
        seen.append(kwargs)
        return result

    monkeypatch.setattr(builders, 'build_uks_integral_inputs', build)
    monkeypatch.setattr(builders, 'run_uks_from_integrals', run)
    monkeypatch.setattr(type(mf), '_configure_jax_cache', lambda self: None)
    monkeypatch.setattr(type(mf), '_put_on_requested_device', lambda self, value: value)
    assert mf.kernel() == result.total_energy
    assert grid_calls == [], 'Ordinary UHF energy must defer quadrature until a reference consumer'
    assert counts == {'integrals': 1, 'scf': 1}
    assert seen[0]['grid_weights'] is inputs.grid_weights
    assert mf.reference is None and mf._scf_inputs is inputs and mf.scf_result is result
    np.testing.assert_array_equal(mf.mo_coeff, np.stack([result.mo_coeff_alpha, result.mo_coeff_beta]))
    np.testing.assert_array_equal(mf.mo_occ, np.stack([result.mo_occ_alpha, result.mo_occ_beta]))
    np.testing.assert_array_equal(mf.mo_energy, np.stack([result.mo_energy_alpha, result.mo_energy_beta]))


@pytest.mark.parametrize('cart', [True, False])
def test_uhf_cached_reference_completes_grid_once_without_resolving_scf(monkeypatch, cart):
    from gradscf.scf import facade
    from gradscf.scf.reference import as_reference

    mf, inputs, result = _uhf_facade_fixture(cart=cart)
    context, ao, _, grid_calls = _grid_fixture(monkeypatch, inputs)
    complete = builders.complete_restricted_response_inputs
    completion_cart = []

    def complete_with_cart(*args, **kwargs):
        completion_cart.append(kwargs.get('cart', True))
        return complete(*args, **kwargs)

    monkeypatch.setattr(builders, 'complete_restricted_response_inputs', complete_with_cart)
    monkeypatch.setattr(facade, 'complete_restricted_response_inputs', complete_with_cart)
    # Seed the solved facade snapshot to isolate completion from the kernel test.
    mf._scf_inputs, mf.scf_result = inputs, result
    mf.e_tot, mf.cycles, mf.converged = result.total_energy, result.cycles, result.converged
    mf.mo_coeff = np.stack([result.mo_coeff_alpha, result.mo_coeff_beta])
    mf.mo_occ = np.stack([result.mo_occ_alpha, result.mo_occ_beta])
    mf.mo_energy = np.stack([result.mo_energy_alpha, result.mo_energy_beta])
    mf._cached_scf_key, mf._cached_orbital_key = mf._scf_signature(), mf._orbital_signature()

    def forbidden(*args, **kwargs):
        raise AssertionError('A solved UHF reference must reuse integrals and SCF result')

    monkeypatch.setattr(builders, 'build_uks_integral_inputs', forbidden)
    monkeypatch.setattr(builders, 'run_uks_from_integrals', forbidden)
    monkeypatch.setattr(type(mf), '_put_on_requested_device', lambda self, value: value)
    reference = as_reference(mf)
    assert len(grid_calls) == 1
    assert completion_cart and all(value is cart for value in completion_cart)
    assert reference.grid.weights is context.grid_weights and reference.ao is ao
    assert mf.scf_result is result
    for name in ('basis', 'overlap', 'hcore', 'eri', 'df_factors', 'direct_basis', 'dipole_integrals'):
        assert getattr(mf._scf_inputs, name) is getattr(inputs, name)
    assert mf._scf_inputs.grid_weights is context.grid_weights
    np.testing.assert_array_equal(reference.rdm1,
        np.stack([result.density_matrix_alpha, result.density_matrix_beta]))
    np.testing.assert_array_equal(reference.mo_coeff, mf.mo_coeff)
    assert reference.mf_energy == result.total_energy and reference.scf_cycles == result.cycles
    assert mf._ensure_reference() is reference and as_reference(mf) is reference
    assert len(grid_calls) == 1


def test_finite_unconverged_uhf_stability_remains_unresolved_without_resolving_scf(monkeypatch):
    from dataclasses import replace
    from gradscf.scf import stability
    from gradscf.scf.reference import as_reference

    mf, inputs, converged_result = _uhf_facade_fixture()
    result = replace(converged_result, converged=False)
    _, _, _, grid_calls = _grid_fixture(monkeypatch, inputs)
    counts = {'integrals': 0, 'scf': 0}

    def build(**kwargs):
        counts['integrals'] += 1
        return inputs

    def run(**kwargs):
        counts['scf'] += 1
        return result

    def problem(coeff, occ, **kwargs):
        np.testing.assert_array_equal(coeff, np.stack([result.mo_coeff_alpha, result.mo_coeff_beta]))
        return 0, None, None, (), np.zeros(0)

    monkeypatch.setattr(builders, 'build_uks_integral_inputs', build)
    monkeypatch.setattr(builders, 'run_uks_from_integrals', run)
    monkeypatch.setattr(stability, '_unrestricted_problem', problem)
    monkeypatch.setattr(type(mf), '_configure_jax_cache', lambda self: None)
    monkeypatch.setattr(type(mf), '_put_on_requested_device', lambda self, value: value)
    mf.kernel()
    assert not mf.converged
    before = len(grid_calls)
    tolerance = mf.conv_tol
    mf.conv_tol = 2*tolerance
    with pytest.raises(RuntimeError, match='SCF|changed'):
        mf.stability()
    assert len(grid_calls) == before
    mf.conv_tol = tolerance
    coefficients = mf.mo_coeff
    mf.mo_coeff = np.array(coefficients, copy=True)
    mf.mo_coeff.flat[0] += .01
    with pytest.raises(RuntimeError, match='(?i)orbital'):
        mf.stability()
    assert len(grid_calls) == before
    mf.mo_coeff = coefficients
    with pytest.raises(RuntimeError, match='converge'):
        as_reference(mf)
    checked = mf.stability()
    assert checked.stable is None and not checked.eigensolver_converged
    assert not mf.converged and not mf.reference.scf_converged
    assert counts == {'integrals': 1, 'scf': 1} and len(grid_calls) == 1
    with pytest.raises(RuntimeError, match='converge'):
        as_reference(mf)
