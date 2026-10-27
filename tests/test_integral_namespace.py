import importlib.util

import numpy as np


def test_integrals_has_canonical_public_namespace():
    assert importlib.util.find_spec("gradscf.integrals") is not None
    from gradscf import integrals
    assert integrals.overlap_matrix.__module__.startswith("gradscf.integrals.")
    assert integrals.eri_tensor.__module__.startswith("gradscf.integrals.")
    top, _ = integrals.prepare_basis("H 0 0 0; H 0 0 .74", "sto-3g")
    assert integrals.make_plan(top).backend == "libcint"


def test_gto_facade_uses_canonical_basis_implementation():
    assert importlib.util.find_spec("gradscf.integrals") is not None
    from gradscf.integrals.basis import CartesianAO, CartesianBasis
    from gradscf.gto import basis as facade
    from gradscf.integrals.basis import prepare_basis
    assert facade.CartesianAO is CartesianAO and facade.CartesianBasis is CartesianBasis
    assert facade.prepare_basis is prepare_basis


def test_retired_integral_modules_are_removed():
    from pathlib import Path
    root = Path("src/gradscf")
    for path in ("data/integrals", "data/basis.py", "data/grid.py", "data/grid_ao.py",
                 "data/pyscf_basis_loader.py", "data/pyscf_basis_snapshot", "scf/inputs.py", "_native"):
        assert not (root/path).exists(), path


def test_reference_integral_still_computes_normalized_overlap():
    assert importlib.util.find_spec("gradscf.integrals") is not None
    from gradscf.integrals import overlap_matrix
    from gradscf.gto.basis import basis_from_spec
    basis = basis_from_spec(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
    matrix = overlap_matrix(basis)
    np.testing.assert_allclose(np.diag(matrix), 1., atol=1e-12)
    assert 0 < matrix[0, 1] < 1
