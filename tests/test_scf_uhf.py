import numpy as np
import pytest

from gradscf import gto, scf


@pytest.fixture
def lih_plus_reference():
    pytest.importorskip("pyscf")
    from pyscf import gto as pyscf_gto, scf as pyscf_scf

    mol = pyscf_gto.M(
        atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", unit="Angstrom",
        charge=1, spin=1, cart=True, verbose=0,
    )
    mf = pyscf_scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    return mol, mf


def test_uhf_integrals_match_pyscf_open_shell(lih_plus_reference):
    assert hasattr(scf, "run_uhf_from_integrals")
    mol, reference = lih_plus_reference
    result = scf.run_uhf_from_integrals(
        overlap=reference.get_ovlp(), hcore=reference.get_hcore(),
        eri=mol.intor("int2e"), nalpha=2, nbeta=1,
        nuclear_repulsion=mol.energy_nuc(),
        config=scf.UHFConfig(max_cycle=100, conv_tol=1e-11, conv_tol_density=1e-9),
    )
    assert isinstance(result, scf.UHFResult)
    assert result.converged
    np.testing.assert_allclose(result.total_energy, reference.e_tot, atol=1e-8, rtol=0)
    density = np.stack([result.density_matrix_alpha, result.density_matrix_beta])
    np.testing.assert_allclose(density, reference.make_rdm1(), atol=2e-6, rtol=0)
    np.testing.assert_allclose(
        np.einsum("sij,ji->s", density, reference.get_ovlp()), [2, 1], atol=1e-10,
    )
    np.testing.assert_allclose(
        [result.mo_occ_alpha.sum(), result.mo_occ_beta.sum()], [2, 1], atol=1e-12,
    )


def test_uhf_basis_one_electron_has_no_self_interaction():
    assert hasattr(scf, "run_uhf")
    pytest.importorskip("pyscf")
    from pyscf import gto as pyscf_gto, scf as pyscf_scf
    from gradscf.integrals.basis import basis_from_pyscf_mol_cart

    mol = pyscf_gto.M(
        atom="H 0 0 0; H 0 0 1.06", basis="sto-3g", charge=1,
        spin=1, cart=True, verbose=0,
    )
    reference = pyscf_scf.UHF(mol).run(conv_tol=1e-12)
    result = scf.run_uhf(
        basis=basis_from_pyscf_mol_cart(mol), nalpha=1, nbeta=0,
        config=scf.UHFConfig(max_cycle=40),
    )
    assert result.converged
    np.testing.assert_allclose(result.total_energy, reference.e_tot, atol=2e-7, rtol=0)
    np.testing.assert_allclose(result.density_matrix_beta, 0, atol=1e-12)


def test_uhf_facade_matches_pyscf(lih_plus_reference):
    assert hasattr(scf, "UHF")
    _, reference = lih_plus_reference
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", charge=1, spin=1)
    mf = scf.UHF(mol, execution_device="cpu", max_cycle=100)
    assert mf.xc == "hf"
    assert mf.run() is mf
    assert mf.converged
    np.testing.assert_allclose(mf.e_tot, reference.e_tot, atol=1e-8, rtol=0)
    assert mf.mo_coeff.shape == (2, 6, 6)
    np.testing.assert_allclose(mf.make_rdm1(), reference.make_rdm1(), atol=2e-6, rtol=0)
    np.testing.assert_allclose(np.sum(mf.mo_occ, axis=1), [2, 1], atol=1e-12)


def test_uhf_facade_rejects_changing_to_dft():
    assert hasattr(scf, "UHF")
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1)
    with pytest.raises(TypeError):
        scf.UHF(mol, xc="pbe")
    mf = scf.UHF(mol)
    mf.xc = "pbe"
    with pytest.raises(ValueError, match="HF"):
        mf.kernel()


def test_uhf_rejects_invalid_occupations():
    assert hasattr(scf, "run_uhf_from_integrals")
    with pytest.raises(ValueError, match="occupation"):
        scf.run_uhf_from_integrals(
            overlap=np.eye(1), hcore=-np.eye(1), eri=np.zeros((1, 1, 1, 1)),
            nalpha=2, nbeta=0, nuclear_repulsion=0.0,
        )


def test_uhf_preserves_broken_spin_symmetry_from_initial_density():
    pytest.importorskip("pyscf")
    from pyscf import gto as pyscf_gto, scf as pyscf_scf

    mol = pyscf_gto.M(
        atom="H 0 0 0; H 0 0 3.0", basis="sto-3g", spin=0, cart=True, verbose=0,
    )
    initial = np.asarray([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
    reference = pyscf_scf.UHF(mol)
    reference.conv_tol = 1e-12
    reference.kernel(dm0=initial)
    assert reference.converged
    result = scf.run_uhf_from_integrals(
        overlap=reference.get_ovlp(), hcore=reference.get_hcore(),
        eri=mol.intor("int2e"), nalpha=1, nbeta=1,
        nuclear_repulsion=mol.energy_nuc(),
        init_density_alpha=initial[0], init_density_beta=initial[1],
    )
    assert result.converged
    np.testing.assert_allclose(result.total_energy, reference.e_tot, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        [result.density_matrix_alpha, result.density_matrix_beta],
        reference.make_rdm1(), atol=2e-6, rtol=0,
    )
    assert np.linalg.norm(result.density_matrix_alpha - result.density_matrix_beta) > 1.0


def test_uhf_density_requires_orbitals():
    mf = scf.UHF(gto.M(atom="H 0 0 0", basis="sto-3g", spin=1))
    with pytest.raises(RuntimeError, match="Run UHF"):
        mf.make_rdm1()
