"""Bounded HF/KS acceptance matrix; run only in the authorized Slurm allocation.

No Cartesian product of all recipes, spin modes, layouts and derivatives is
implied. Registry metadata, molecular support and local derivatives are separate
contracts. Molecular energies use Ha, coordinates Angstrom, and float64.
"""
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from gradscf import dft, gto, scf
from gradscf.dft import xc
from gradscf.dft.xc.registry import ALIASES, RECIPES


def test_every_registered_recipe_and_alias_retains_complete_requirements(tmp_path):
    """Enumerate everything, without calling metadata a numerical validation."""
    records = []
    assert {recipe.name for recipe in xc.list_functional_recipes()} == set(RECIPES)
    for name in sorted(set(RECIPES) | set(ALIASES)):
        recipe = xc.parse_recipe(name)
        expected = RECIPES[ALIASES.get(name, name)]
        assert recipe == expected
        metadata = xc.info(name)
        assert metadata['recipe'] == expected
        extras = [label for label in ('dispersion', 'nlc', 'pt2')
                  if getattr(expected, label) is not None]
        assert set(metadata['extra_requirements']) == set(extras)
        if extras:
            with pytest.raises(NotImplementedError, match='requires'):
                recipe.require_complete_support()
        else:
            recipe.require_complete_support()
        if recipe.needs_laplacian:
            assert not metadata['backend_available']
            assert 'laplacian' in metadata['unsupported_features']
        records.append(dict(name=name, canonical=recipe.name, recipe=asdict(recipe),
            local_channels=metadata['local_channels'],
            local_backend_available=metadata['backend_available'],
            extra_requirements=extras,
            unsupported_features=metadata['unsupported_features'],
            molecular_validation='not established by inventory'))
    path = tmp_path / 'functional-capabilities.json'
    path.write_text(json.dumps(records, indent=2) + '\n')
    print(f'Complete registry inventory: {len(RECIPES)} recipes, '
          f'{len(ALIASES)} aliases; evidence={path}', flush=True)


def _molecule(method):
    if method.startswith('R'):
        return gto.M(atom='O 0 0 0; H 0 .76 .58; H 0 -.76 .58',
                     basis='sto-3g', cart=True)
    return gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g',
                 charge=1, spin=1, cart=True)


def _assert_stationary(mf):
    assert mf.converged
    result = mf.scf_result
    if result is None:
        # UKS stores its solved spin matrices in the unrestricted reference.
        # Read that actual Fock; do not manufacture a diagonal orbital Fock.
        result = mf.reference
        assert result is not None and result.scf_converged
        fock = np.asarray(result.fock_matrix)
        cycles = result.scf_cycles
    else:
        assert result.converged
        fock = (np.stack([result.fock_matrix_alpha, result.fock_matrix_beta])
                if mf.mo_coeff.ndim == 3 else np.asarray(result.fock_matrix))
        cycles = result.cycles
    coefficients, occupations = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
    assert fock.shape == coefficients.shape
    fmo = coefficients.conj().swapaxes(-1, -2) @ fock @ coefficients
    difference = occupations[..., None, :] - occupations[..., :, None]
    # Independent occupation-weighted orbital residual, not the converged flag.
    residual = np.linalg.norm(fmo*difference) / np.sqrt(2.)
    assert np.isfinite(mf.e_tot) and np.isfinite(fock).all()
    assert residual <= 1e-7, f'orbital residual {residual} Ha'
    assert cycles is not None and int(cycles) > 0
    return residual, int(cycles)


def _pyscf_reference(mol, method, spec, backend, inputs):
    from pyscf import dft as pdft, gto as pgto, scf as pscf

    pmol = pgto.M(atom=mol.atom, basis=mol.basis, unit=mol.unit,
                  charge=mol.charge, spin=mol.spin, cart=mol.cart, verbose=0)
    # The generalized facade uses a collinear molecular fixture here; complex
    # spin mixing has independent dense J/K and PySCF tests in other files.
    reference_method = method.replace('G', 'U', 1)
    reference = getattr(pdft if method.endswith('KS') else pscf, reference_method)(pmol)
    if backend == 'df':
        reference = reference.density_fit(auxbasis='def2-universal-jkfit')
    if method.endswith('KS'):
        reference.xc = {'lc-wpbe': 'LC_WPBE'}.get(spec, spec)
        if spec == 'lc-wpbe':
            # Libxc uses (omega, LR alpha, SR-minus-LR beta); GradSCF uses
            # (omega, LR, SR). Both must retain omega=0.4 Bohr^-1, SR=0, LR=1.
            omega, alpha, beta = pdft.libxc.rsh_coeff(reference.xc)
            np.testing.assert_allclose((omega, alpha, beta), (.4, 1., -1.),
                                       atol=1e-14, rtol=0)
            np.testing.assert_allclose(
                xc.parse_recipe(spec).to_range_separated_hybrid_coefficients(),
                (omega, alpha, alpha+beta), atol=1e-14, rtol=0)
        # Use the identical full grid; differences in pruning are not an XC gate.
        reference.grids.coords = np.asarray(inputs.coords)
        reference.grids.weights = np.asarray(inputs.grid_weights)
        reference.small_rho_cutoff = 0.
    reference.init_guess = '1e'
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    reference.max_cycle = 150
    reference.kernel()
    assert reference.converged
    return reference


@pytest.mark.parametrize('method', ['RHF', 'UHF', 'GHF', 'RKS', 'UKS', 'GKS'])
@pytest.mark.parametrize('backend', ['direct', 'df'])
def test_six_facades_conventional_and_true_ri_match_independent_reference(method, backend):
    """True auxiliary RI is compared with the same approximation in PySCF."""
    mol = _molecule(method)
    options = dict(conv_tol=1e-11, conv_tol_density=1e-9, conv_tol_grad=1e-8,
                   max_cycle=150, max_memory=1024., jk_backend=backend,
                   init_guess='hcore', direct_scf_tol=1e-12, direct_scf_workers=2)
    if backend == 'df':
        options['auxbasis'] = 'def2-universal-jkfit'
    if method.endswith('KS'):
        options.update(xc='pbe0', grids_level=0)
    mf = getattr(dft if method.endswith('KS') else scf, method)(mol, **options).run()
    residual, cycles = _assert_stationary(mf)
    inputs = mf._scf_inputs
    if inputs is None:
        assert method == 'UKS'
        state = mf.reference
        inputs = SimpleNamespace(overlap=state.overlap_matrix, eri=state.rep_tensor,
            df_factors=state.df_factors, direct_basis=state.direct_basis,
            coords=state.grid.coords, grid_weights=state.grid.weights)
    if backend == 'df':
        assert inputs.df_factors is not None and inputs.df_factors.ndim == 3
        assert inputs.df_factors.shape[0] > inputs.overlap.shape[0]
        assert inputs.eri is None or inputs.eri.size == 0
        assert inputs.direct_basis is None
    else:
        assert inputs.direct_basis is not None and inputs.df_factors is None
        assert inputs.eri is None or inputs.eri.size == 0
    reference = _pyscf_reference(mol, method, 'pbe0', backend, inputs)
    np.testing.assert_allclose(mf.e_tot, reference.e_tot, atol=1e-7, rtol=0)
    print(dict(method=method, backend=backend, cycles=cycles,
               energy_hartree=float(mf.e_tot), reference_hartree=float(reference.e_tot),
               energy_error_hartree=float(mf.e_tot-reference.e_tot),
               orbital_residual_hartree=float(residual)), flush=True)


@pytest.mark.parametrize('spec', ['svwn', 'pbe', 'r2scan', 'm06-2x',
                                 'pbe0', 'cam-b3lyp', 'lc-wpbe', 'hse06'])
def test_molecular_category_representatives_match_independent_reference(spec):
    """One direct RKS water per executable category, with full exchange kept."""
    mol = _molecule('RKS')
    mf = dft.RKS(mol, xc=spec, grids_level=0, jk_backend='direct',
                 direct_scf_tol=1e-12, direct_scf_workers=2, init_guess='hcore',
                 conv_tol=1e-11, conv_tol_density=1e-9, conv_tol_grad=1e-8,
                 max_cycle=150, max_memory=1024.).run()
    residual, cycles = _assert_stationary(mf)
    reference = _pyscf_reference(mol, 'RKS', spec, 'direct', mf._scf_inputs)
    np.testing.assert_allclose(mf.e_tot, reference.e_tot, atol=1e-7, rtol=0)
    print(dict(method='RKS', xc=spec, backend='direct', cycles=cycles,
               energy_hartree=float(mf.e_tot), reference_hartree=float(reference.e_tot),
               energy_error_hartree=float(mf.e_tot-reference.e_tot),
               orbital_residual_hartree=float(residual)), flush=True)


@pytest.mark.parametrize('spec,reason', [
    ('scanl', 'Laplacian'), ('r2scanl', 'Laplacian'),
    ('b97d', 'dispersion'), ('wb97x_d3', 'dispersion'),
    ('b97m_v', 'NLC'), ('wb97x_v', 'NLC'), ('wb97m_v', 'NLC'),
    ('scan_rvv10', 'NLC'), ('b2plyp', 'PT2'),
])
def test_incomplete_molecular_categories_fail_before_numerical_preparation(spec, reason, monkeypatch):
    from gradscf.scf.inputs import assembly

    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported physics must be rejected before preparation')

    monkeypatch.setattr(assembly, '_prepare_basis_grid_context', forbidden)
    with pytest.raises(NotImplementedError, match=reason):
        dft.RKS(_molecule('RKS'), xc=spec).kernel()


@pytest.mark.parametrize('options,reason', [
    (dict(xc='r2scan'), 'meta-GGA'),
    (dict(xc='pbe', collinear='ncol'), 'LDA'),
    (dict(xc='cam-b3lyp', jk_backend='df', auxbasis='def2-universal-jkfit'), 'RI exchange'),
])
def test_generalized_unsupported_modes_are_not_counted_as_coverage(options, reason):
    with pytest.raises(NotImplementedError, match=reason):
        dft.GKS(_molecule('GKS'), **options).kernel()
