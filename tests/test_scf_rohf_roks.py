import numpy as np
import pytest

from gradscf import dft, gto, scf


def _reference(xc="hf", spin=1):
    pytest.importorskip("pyscf")
    from pyscf import dft as pdft, gto as pgto, scf as pscf

    mol = pgto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", charge=1,
                 spin=spin, cart=True, verbose=0)
    mf = pscf.ROHF(mol) if xc == "hf" else pdft.ROKS(mol, xc=xc)
    if xc != "hf":
        mf.grids.level = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged
    return mol, mf


def _assert_reference(result, mol, ref, energy_tol=1e-8, swap_spins=False):
    assert result.converged
    np.testing.assert_allclose(result.total_energy, ref.e_tot, atol=energy_tol, rtol=0)
    assert result.mo_coeff.shape == (mol.nao, mol.nao)
    assert sorted(np.asarray(result.mo_occ).tolist()) == [0., 0., 0., 0., 1., 2.]
    dm = np.stack([result.density_matrix_alpha, result.density_matrix_beta])
    reference_dm = ref.make_rdm1()[::-1] if swap_spins else ref.make_rdm1()
    np.testing.assert_allclose(dm, reference_dm, atol=3e-6, rtol=0)
    s = ref.get_ovlp()
    np.testing.assert_allclose(result.mo_coeff.T @ s @ result.mo_coeff, np.eye(mol.nao), atol=1e-10)
    counts = mol.nelec[::-1] if swap_spins else mol.nelec
    np.testing.assert_allclose(np.einsum("sij,ji->s", dm, s), counts, atol=1e-10)
    major, minor = dm[::-1] if swap_spins else dm
    # The doubly occupied subspace is contained in the majority-spin space.
    np.testing.assert_allclose(major @ s @ minor, minor, atol=1e-10)
    assert result.orbital_gradient_norm < 1e-7


@pytest.mark.parametrize("spin", [1, -1])
def test_rohf_shared_orbitals_match_pyscf(spin):
    assert hasattr(scf, "run_rohf_from_integrals")
    mol, ref = _reference()
    nalpha, nbeta = mol.nelec if spin > 0 else mol.nelec[::-1]
    result = scf.run_rohf_from_integrals(
        overlap=ref.get_ovlp(), hcore=ref.get_hcore(), eri=mol.intor("int2e"),
        nalpha=nalpha, nbeta=nbeta, nuclear_repulsion=mol.energy_nuc(),
        config=scf.ROHFConfig(max_cycle=100, conv_tol_density=1e-9),
    )
    _assert_reference(result, mol, ref, swap_spins=spin < 0)


@pytest.mark.parametrize("xc", ["lda_x", "pbe", "b3lyp"])
def test_roks_matches_pyscf_on_identical_grid(xc):
    assert hasattr(scf, "run_roks_from_integrals")
    pytest.importorskip("jax_xc")
    from pyscf.dft import numint

    mol, ref = _reference(xc)
    ao = numint.eval_ao(mol, ref.grids.coords, deriv=1)
    result = scf.run_roks_from_integrals(
        overlap=ref.get_ovlp(), hcore=ref.get_hcore(), eri=mol.intor("int2e"),
        nalpha=2, nbeta=1, nuclear_repulsion=mol.energy_nuc(),
        ao=ao[0], ao_deriv1=ao, grid_weights=ref.grids.weights,
        config=scf.ROKSConfig(xc_spec=xc, max_cycle=100, conv_tol_density=1e-9),
    )
    _assert_reference(result, mol, ref, energy_tol=2e-7)


def test_rohf_closed_shell_limit_matches_rhf():
    assert hasattr(scf, "run_rohf_from_integrals")
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto, scf as pscf

    mol = pgto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True, verbose=0)
    ref = pscf.RHF(mol).run(conv_tol=1e-12)
    result = scf.run_rohf_from_integrals(
        overlap=ref.get_ovlp(), hcore=ref.get_hcore(), eri=mol.intor("int2e"),
        nalpha=1, nbeta=1, nuclear_repulsion=mol.energy_nuc(),
    )
    assert result.converged
    np.testing.assert_allclose(result.total_energy, ref.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(result.density_matrix, ref.make_rdm1(), atol=1e-8)


def test_rohf_and_roks_facades():
    assert hasattr(scf, "ROHF") and hasattr(dft, "ROKS")
    assert dft.ROKS is scf.ROKS
    mol, ref = _reference()
    local_mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", charge=1, spin=1)
    mf = scf.ROHF(local_mol, max_cycle=100, execution_device="cpu").run()
    assert mf.converged
    np.testing.assert_allclose(mf.e_tot, ref.e_tot, atol=1e-8, rtol=0)
    assert mf.mo_coeff.shape == (mol.nao, mol.nao)
    np.testing.assert_allclose(mf.make_rdm1(), ref.make_rdm1(), atol=3e-6, rtol=0)
    with pytest.raises(NotImplementedError, match="restricted.open.shell"):
        mf.TDA()
    with pytest.raises(NotImplementedError, match="restricted.open.shell"):
        mf.TDDFT()
    with pytest.raises(TypeError):
        scf.ROHF(local_mol, xc="pbe")
    mf.xc = "pbe"
    with pytest.raises(ValueError, match="HF"):
        mf.kernel()


def test_roks_facade_pbe_matches_reference():
    assert hasattr(dft, "ROKS")
    pytest.importorskip("jax_xc")
    _, ref = _reference("pbe")
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", charge=1, spin=1)
    mf = dft.ROKS(mol, xc="pbe", grids_level=0, max_cycle=100).run()
    assert mf.converged
    np.testing.assert_allclose(mf.e_tot, ref.e_tot, atol=2e-7, rtol=0)


def test_rohf_nonconvergence_is_not_reported_as_success():
    assert hasattr(scf, "run_rohf_from_integrals")
    mol, ref = _reference()
    result = scf.run_rohf_from_integrals(
        overlap=ref.get_ovlp(), hcore=ref.get_hcore(), eri=mol.intor("int2e"),
        nalpha=2, nbeta=1, nuclear_repulsion=mol.energy_nuc(),
        config=scf.ROHFConfig(max_cycle=1, conv_tol=1e-14, conv_tol_density=1e-14),
    )
    assert not result.converged
    assert result.cycles == 1


def test_rohf_rejects_invalid_occupations():
    assert hasattr(scf, "run_rohf_from_integrals")
    with pytest.raises(ValueError, match="occupation"):
        scf.run_rohf_from_integrals(
            overlap=np.eye(1), hcore=-np.eye(1), eri=np.zeros((1, 1, 1, 1)),
            nalpha=2, nbeta=0, nuclear_repulsion=0.,
        )


def test_roothaan_fock_and_orbital_gradient_match_pyscf():
    pytest.importorskip("pyscf")
    from pyscf.scf import rohf
    from gradscf.scf.roks import roothaan_fock, _ro_gradient

    mol, ref = _reference()
    coeff = ref.mo_coeff
    dm = ref.make_rdm1()
    rng = np.random.default_rng(17)
    fock = rng.normal(size=(2, mol.nao, mol.nao))
    fock = fock + fock.transpose(0, 2, 1)
    effective = roothaan_fock(fock[0], fock[1], dm[0], dm[1], ref.get_ovlp())
    np.testing.assert_allclose(effective, rohf.get_roothaan_fock(fock, dm, ref.get_ovlp()), atol=1e-12)
    occ_spin = np.stack([ref.mo_occ > 0, ref.mo_occ == 2]).astype(float)
    np.testing.assert_allclose(
        _ro_gradient(coeff, occ_spin, fock), np.linalg.norm(rohf.get_grad(coeff, ref.mo_occ, fock)),
        atol=1e-12,
    )


def test_rohf_basis_one_electron():
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto, scf as pscf
    from pyscf_adapters import basis_from_pyscf_mol_cart

    mol = pgto.M(atom="H 0 0 0; H 0 0 1.06", basis="sto-3g", charge=1,
                 spin=1, cart=True, verbose=0)
    ref = pscf.ROHF(mol).run()
    result = scf.run_rohf(basis=basis_from_pyscf_mol_cart(mol), nalpha=1, nbeta=0)
    assert result.converged
    np.testing.assert_allclose(result.total_energy, ref.e_tot, atol=2e-7, rtol=0)
    np.testing.assert_allclose(result.density_matrix_beta, 0, atol=1e-12)


def test_ro_occupations_use_majority_energies_for_open_orbitals():
    from gradscf.scf.roks import _ro_occupations

    # The open orbital is not necessarily the next effective-Fock eigenvalue.
    result = _ro_occupations(np.array([-3., -2., -1., 1.]),
                             np.array([-3., 1., -2., 0.]), ncore=1, nopen=1)
    np.testing.assert_array_equal(result, [2., 0., 1., 0.])


def test_rohf_energy_gradient_with_respect_to_hcore():
    import jax
    import jax.numpy as jnp

    mol, ref = _reference()
    h = jnp.asarray(ref.get_hcore())
    kwargs = dict(overlap=ref.get_ovlp(), eri=mol.intor("int2e"), nalpha=2, nbeta=1,
                  nuclear_repulsion=mol.energy_nuc(),
                  config=scf.ROHFConfig(max_cycle=100, conv_tol_density=1e-10))
    result = scf.run_rohf_from_integrals(hcore=h, **kwargs)
    derivative = jax.grad(lambda hcore: scf.run_rohf_from_integrals(hcore=hcore, **kwargs).total_energy)(h)
    # At stationarity, dE/dh_pq equals the spin-summed AO density.
    np.testing.assert_allclose(derivative, result.density_matrix, atol=2e-6, rtol=0)


def test_roks_closed_shell_limit_matches_pyscf_rks():
    pytest.importorskip("jax_xc")
    from pyscf import dft as pdft, gto as pgto

    mol = pgto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True, verbose=0)
    ref = pdft.RKS(mol, xc="pbe")
    ref.grids.level = 0
    ref.kernel()
    ao = pdft.numint.eval_ao(mol, ref.grids.coords, deriv=1)
    result = scf.run_roks_from_integrals(
        overlap=ref.get_ovlp(), hcore=ref.get_hcore(), eri=mol.intor("int2e"),
        nalpha=1, nbeta=1, nuclear_repulsion=mol.energy_nuc(), ao=ao[0],
        ao_deriv1=ao, grid_weights=ref.grids.weights, config=scf.ROKSConfig(xc_spec="pbe"),
    )
    assert result.converged
    np.testing.assert_allclose(result.total_energy, ref.e_tot, atol=2e-7, rtol=0)


def test_roks_external_response_entry_rejects_reference():
    from gradscf import tdscf

    mf = scf.ROHF(gto.M(atom="H 0 0 0", basis="sto-3g", spin=1))
    mf.e_tot = -0.4
    with pytest.raises(NotImplementedError, match="restricted.open.shell"):
        _ = tdscf.TDA(mf).reference


def test_rohf_triplet_with_two_open_orbitals():
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto, scf as pscf

    mol = pgto.M(atom="O 0 0 0; O 0 0 1.2", basis="sto-3g", spin=2, cart=True, verbose=0)
    ref = pscf.ROHF(mol).run(conv_tol=1e-12)
    assert ref.converged
    result = scf.run_rohf_from_integrals(
        overlap=ref.get_ovlp(), hcore=ref.get_hcore(), eri=mol.intor("int2e"),
        nalpha=9, nbeta=7, nuclear_repulsion=mol.energy_nuc(),
        config=scf.ROHFConfig(max_cycle=120),
    )
    assert result.converged
    assert np.count_nonzero(np.asarray(result.mo_occ) == 1) == 2
    np.testing.assert_allclose(result.total_energy, ref.e_tot, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        result.density_matrix_alpha @ ref.get_ovlp() @ result.density_matrix_beta,
        result.density_matrix_beta, atol=1e-10,
    )
