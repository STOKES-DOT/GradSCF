"""Retiring Gaussian recurrence metadata keeps the live AO/grid payload."""
import ast
from dataclasses import fields
import inspect
from pathlib import Path

import numpy as np


def test_cartesian_basis_has_no_retired_quartet_metadata_or_flags():
    from gradscf.integrals import basis as module
    from gradscf.integrals.basis import CartesianBasis, basis_from_spec, basis_from_molecule_spec

    retired = {'quartet_groups', 'shell_quartet_groups', 'precompute_eri_groups'}
    names = {field.name for field in fields(CartesianBasis)}
    assert not names & retired, f'Retired recurrence metadata remains: {names & retired}'
    for factory in (CartesianBasis, basis_from_spec, basis_from_molecule_spec):
        assert 'precompute_eri_groups' not in inspect.signature(factory).parameters
    assert not hasattr(module, 'QuartetBatchGroup')
    assert not hasattr(module, 'ShellQuartetBatchGroup')
    path = Path(module.__file__).with_name('cartesian.py')
    tree = ast.parse(path.read_text())
    assert not any(isinstance(node, ast.Name) and node.id == 'precompute_eri_groups'
                   for node in ast.walk(tree))
    assert not any(isinstance(node, ast.ClassDef) and
                   node.name in {'QuartetBatchGroup', 'ShellQuartetBatchGroup'}
                   for node in ast.walk(tree))


def test_cartesian_basis_retains_ao_grid_arrays_and_shell_mapping():
    from gradscf.integrals.basis import basis_from_spec

    basis = basis_from_spec('H 0 0 0; H .1 .2 .74', basis='3-21g')
    assert basis.nao == 4 and len(basis.shells) == 4
    assert basis.ao_centers.shape == (4, 3)
    assert basis.ao_exponents_padded.shape[0] == 4
    assert basis.ao_coefficients_padded.shape == basis.ao_exponents_padded.shape
    assert basis.shell_centers.shape == (4, 3)
    assert basis.shell_exponents_padded.shape == basis.shell_coefficients_padded.shape
    assert basis.shell_ao_indices_padded.shape[0] == 4
    np.testing.assert_array_equal(np.concatenate([shell.ao_indices for shell in basis.shells]), np.arange(4))
    for index, ao in enumerate(basis.aos):
        count = len(ao.exponents)
        np.testing.assert_allclose(basis.ao_centers[index], ao.center, atol=0, rtol=0)
        np.testing.assert_allclose(basis.ao_exponents_padded[index, :count], ao.exponents, atol=0, rtol=0)
        np.testing.assert_allclose(basis.ao_coefficients_padded[index, :count], ao.coefficients, atol=0, rtol=0)
