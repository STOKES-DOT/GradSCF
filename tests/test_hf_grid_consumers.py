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
