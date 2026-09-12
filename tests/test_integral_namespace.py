import importlib.util

import numpy as np


def test_integrals_has_canonical_public_namespace():
    assert importlib.util.find_spec("gradscf.integrals") is not None
    from gradscf import integrals
    from gradscf.integrals.backends import jax_reference
    assert integrals.overlap_matrix is jax_reference.overlap_matrix
    assert integrals.eri_tensor is jax_reference.eri_tensor


def test_basis_and_legacy_imports_share_objects():
    assert importlib.util.find_spec("gradscf.integrals") is not None
    from gradscf.integrals.basis import CartesianAO, CartesianBasis
    from gradscf.data.basis import CartesianAO as OldAO, CartesianBasis as OldBasis
    from gradscf.data.integrals import overlap_matrix as old_overlap
    from gradscf.integrals import overlap_matrix
    assert CartesianAO is OldAO and CartesianBasis is OldBasis
    assert overlap_matrix is old_overlap


def test_reference_integral_still_computes_normalized_overlap():
    assert importlib.util.find_spec("gradscf.integrals") is not None
    from gradscf.integrals import overlap_matrix
    from gradscf.gto.basis import basis_from_spec
    basis = basis_from_spec(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
    matrix = overlap_matrix(basis)
    np.testing.assert_allclose(np.diag(matrix), 1., atol=1e-12)
    assert 0 < matrix[0, 1] < 1
