import numpy as np
import pytest

from gradscf import dft, gto, scf


def _h2():
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto
    return pgto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", cart=True, verbose=0)


def _spin_hcore(mol):
    from pyscf import scf as pscf
    h = pscf.GHF(mol).get_hcore().astype(complex)
    rng = np.random.default_rng(71)
    field = rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape)
    return h + 0.12 * (field + field.conj().T)


def _reference(xc="hf", collinear="col"):
    from pyscf import dft as pdft, scf as pscf
    mol = _h2()
    mf = pscf.GHF(mol) if xc == "hf" else pdft.GKS(mol, xc=xc)
    if xc != "hf":
        mf.collinear = collinear
        mf.grids.level = 0
    h = _spin_hcore(mol)
    mf.get_hcore = lambda *args: h
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 150
    mf.kernel()
    assert mf.converged
    return mol, mf, h


def _check_solution(result, mol, reference, atol=1e-8):
    assert result.converged
    assert result.cycles > 1
    np.testing.assert_allclose(result.total_energy, reference.e_tot, atol=atol, rtol=0)
    dm = np.asarray(result.density_matrix)
    s = reference.get_ovlp()
    np.testing.assert_allclose(dm, reference.make_rdm1(), atol=3e-6, rtol=0)
    np.testing.assert_allclose(dm, dm.conj().T, atol=1e-12)
    np.testing.assert_allclose(dm @ s @ dm, dm, atol=1e-10)
    np.testing.assert_allclose(np.trace(dm @ s), mol.nelectron, atol=1e-10)
    np.testing.assert_allclose(result.mo_coeff.conj().T @ s @ result.mo_coeff,
                               np.eye(2 * mol.nao), atol=1e-10)
    np.testing.assert_allclose(result.fock_matrix, result.fock_matrix.conj().T, atol=1e-12)
    assert np.linalg.norm(dm[:mol.nao, mol.nao:].imag) > 1e-3
    assert result.mo_coeff.shape == (2 * mol.nao, 2 * mol.nao)
    assert set(np.asarray(result.mo_occ)) <= {0., 1.}


def test_generalized_namespace_exports():
    assert hasattr(scf, "GHF") and hasattr(dft, "GKS")
    assert scf.GKS is dft.GKS


def test_complex_generalized_jk_matches_pyscf():
    assert hasattr(scf, "GHF")
    from pyscf import scf as pscf
    from gradscf.scf.gks import generalized_jk

    mol = _h2()
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    dm = a @ a.conj().T
    expected_j, expected_k = pscf.GHF(mol).get_jk(dm=dm, hermi=1)
    j, k = generalized_jk(mol.intor("int2e"), dm)
    np.testing.assert_allclose(j, expected_j, atol=1e-11, rtol=0)
    np.testing.assert_allclose(k, expected_k, atol=1e-11, rtol=0)
    assert np.linalg.norm(np.asarray(k)[:2, 2:].imag) > .1


def test_ghf_complex_spinor_hamiltonian_matches_pyscf():
    assert hasattr(scf, "run_ghf_from_integrals")
    mol, ref, h = _reference()
    result = scf.run_ghf_from_integrals(
        overlap=mol.intor("int1e_ovlp"), hcore=h, eri=mol.intor("int2e"),
        nelectron=mol.nelectron, nuclear_repulsion=mol.energy_nuc(),
        config=scf.GHFConfig(max_cycle=150, conv_tol_density=1e-9),
    )
    _check_solution(result, mol, ref)


@pytest.mark.parametrize("xc,collinear", [("pbe", "col"), ("b3lyp", "col"), ("lda_x", "ncol")])
def test_gks_matches_pyscf_with_complex_spin_mixing(xc, collinear):
    assert hasattr(scf, "run_gks_from_integrals")
    pytest.importorskip("jax_xc")
    from pyscf.dft import numint

    mol, ref, h = _reference(xc, collinear)
    ao = numint.eval_ao(mol, ref.grids.coords, deriv=1)
    result = scf.run_gks_from_integrals(
        overlap=mol.intor("int1e_ovlp"), hcore=h, eri=mol.intor("int2e"),
        nelectron=mol.nelectron, nuclear_repulsion=mol.energy_nuc(),
        ao=ao[0], ao_deriv1=ao, grid_weights=ref.grids.weights,
        config=scf.GKSConfig(xc_spec=xc, collinear=collinear, max_cycle=150,
                             conv_tol_density=1e-9),
    )
    _check_solution(result, mol, ref, atol=2e-7)


def _rotation(nao):
    theta, phase = .61, .83
    u = np.array([[np.cos(theta), -np.exp(-1j * phase) * np.sin(theta)],
                  [np.exp(1j * phase) * np.sin(theta), np.cos(theta)]])
    return np.kron(u, np.eye(nao))


@pytest.mark.parametrize("xc", ["hf", "lda_x"])
def test_generalized_energy_and_fock_are_spin_rotation_covariant(xc):
    assert hasattr(scf, "GKS")
    if xc != "hf":
        pytest.importorskip("jax_xc")
    from pyscf import dft as pdft, scf as pscf
    from gradscf.scf.gks import generalized_energy_and_fock

    mol = _h2()
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    q, _ = np.linalg.qr(a)
    dm = q @ q.conj().T
    grids = pdft.gen_grid.Grids(mol)
    grids.level = 0
    grids.build()
    ao = pdft.numint.eval_ao(mol, grids.coords, deriv=1)
    kwargs = dict(hcore=pscf.GHF(mol).get_hcore(), eri=mol.intor("int2e"),
                  nuclear_repulsion=mol.energy_nuc(), ao=ao[0], ao_deriv1=ao,
                  grid_weights=grids.weights,
                  config=scf.GKSConfig(xc_spec=xc, collinear="ncol"))
    e, _, f = generalized_energy_and_fock(density=dm, **kwargs)
    u = _rotation(mol.nao)
    rotated = u @ dm @ u.conj().T
    er, _, fr = generalized_energy_and_fock(density=rotated, **kwargs)
    np.testing.assert_allclose(er, e, atol=1e-10, rtol=0)
    np.testing.assert_allclose(fr, u @ f @ u.conj().T, atol=1e-9, rtol=0)


def test_ghf_facade_accepts_complex_initial_density():
    assert hasattr(scf, "GHF")
    from pyscf import gto as pgto, scf as pscf

    atom = "Li 0 0 0; H 0 0 1.6"
    pmol = pgto.M(atom=atom, basis="sto-3g", spin=1, charge=1, cart=True, verbose=0)
    ref = pscf.UHF(pmol).run(conv_tol=1e-12).to_ghf()
    u = _rotation(pmol.nao)
    initial = u @ ref.make_rdm1() @ u.conj().T
    mol = gto.M(atom=atom, basis="sto-3g", spin=1, charge=1)
    mf = scf.GHF(mol, execution_device="cpu", max_cycle=100)
    energy = mf.kernel(dm0=initial)
    assert mf.converged
    np.testing.assert_allclose(energy, ref.e_tot, atol=1e-8, rtol=0)
    np.testing.assert_allclose(mf.make_rdm1(), initial, atol=3e-6, rtol=0)
    assert np.linalg.norm(np.asarray(mf.make_rdm1())[:pmol.nao, pmol.nao:].imag) > 1e-3
    with pytest.raises(TypeError):
        scf.GHF(mol, xc="pbe")
    mf.xc = "pbe"
    with pytest.raises(ValueError, match="HF"):
        mf.kernel()


def test_gks_facade_collinear_pbe():
    assert hasattr(dft, "GKS")
    pytest.importorskip("jax_xc")
    from pyscf import dft as pdft

    pmol = _h2()
    ref = pdft.GKS(pmol, xc="pbe")
    ref.grids.level = 0
    ref.kernel()
    mf = dft.GKS(gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g"), xc="pbe").run()
    assert mf.converged and ref.converged
    np.testing.assert_allclose(mf.e_tot, ref.e_tot, atol=2e-7, rtol=0)


def test_generalized_rejects_unsupported_xc_modes():
    assert hasattr(scf, "run_gks_from_integrals")
    kwargs = dict(overlap=np.eye(1), hcore=-np.eye(1), eri=np.zeros((1, 1, 1, 1)),
                  nelectron=1, nuclear_repulsion=0., ao=np.ones((1, 1)),
                  ao_deriv1=np.zeros((4, 1, 1)), grid_weights=np.ones(1))
    for cfg in [scf.GKSConfig(xc_spec="pbe", collinear="ncol"),
                scf.GKSConfig(xc_spec="lda_x", collinear="mcol")]:
        with pytest.raises(NotImplementedError):
            scf.run_gks_from_integrals(**kwargs, config=cfg)
    # An unavailable functional can be rejected by the XC registry before the
    # generalized solver's meta-GGA guard is reached.
    with pytest.raises((NotImplementedError, KeyError, ValueError)):
        scf.run_gks_from_integrals(**kwargs, config=scf.GKSConfig(xc_spec="mgga_x_scan"))


def test_ghf_nonconvergence_and_invalid_input():
    assert hasattr(scf, "run_ghf_from_integrals")
    mol = _h2()
    kwargs = dict(overlap=mol.intor("int1e_ovlp"), hcore=_spin_hcore(mol),
                  eri=mol.intor("int2e"), nelectron=2, nuclear_repulsion=mol.energy_nuc())
    result = scf.run_ghf_from_integrals(**kwargs, config=scf.GHFConfig(max_cycle=1))
    assert not result.converged
    with pytest.raises(ValueError, match="Hermitian"):
        scf.run_ghf_from_integrals(**kwargs, init_density=np.triu(np.ones((4, 4))))
    with pytest.raises(ValueError, match="electron"):
        scf.run_ghf_from_integrals(**dict(kwargs, nelectron=5))


def test_ghf_jax_basis_one_electron_spinor():
    from pyscf import gto as pgto, scf as pscf
    from gradscf.integrals.basis import basis_from_pyscf_mol_cart

    mol = pgto.M(atom="H 0 0 0; H 0 0 1.06", basis="sto-3g", charge=1,
                 spin=1, cart=True, verbose=0)
    ref = pscf.GHF(mol).run()
    u = _rotation(mol.nao)
    result = scf.run_ghf(
        basis=basis_from_pyscf_mol_cart(mol), nelectron=1,
        init_density=u @ ref.make_rdm1() @ u.conj().T,
    )
    assert result.converged
    np.testing.assert_allclose(result.total_energy, ref.e_tot, atol=2e-7, rtol=0)
    assert np.linalg.norm(np.asarray(result.density_matrix)[:2, 2:].imag) > 1e-3


def test_noncollinear_lda_zero_magnetization_is_finite():
    pytest.importorskip("jax_xc")
    from pyscf import dft as pdft, scf as pscf
    from gradscf.scf.gks import generalized_energy_and_fock

    mol = _h2()
    grids = pdft.gen_grid.Grids(mol)
    grids.level = 0
    grids.build()
    ao = pdft.numint.eval_ao(mol, grids.coords, deriv=1)
    dm = np.eye(4, dtype=complex) * .25
    kwargs = dict(density=dm, hcore=pscf.GHF(mol).get_hcore(), eri=mol.intor("int2e"),
                  nuclear_repulsion=mol.energy_nuc(), ao=ao[0], ao_deriv1=ao, grid_weights=grids.weights)
    ec, _, fc = generalized_energy_and_fock(**kwargs, config=scf.GKSConfig(xc_spec="lda_x"))
    en, _, fn = generalized_energy_and_fock(**kwargs, config=scf.GKSConfig(xc_spec="lda_x", collinear="ncol"))
    assert np.isfinite(en) and np.all(np.isfinite(fn))
    np.testing.assert_allclose(en, ec, atol=1e-12, rtol=0)
    np.testing.assert_allclose(fn, fc, atol=1e-12, rtol=0)


def test_generalized_response_is_explicitly_unsupported():
    from gradscf import tdscf

    mf = scf.GHF(gto.M(atom="H 0 0 0", basis="sto-3g", spin=1))
    with pytest.raises(RuntimeError, match="Run generalized"):
        mf.make_rdm1()
    with pytest.raises(NotImplementedError, match="Generalized spinor"):
        mf.TDA()
    with pytest.raises(NotImplementedError, match="Generalized spinor"):
        _ = tdscf.TDDFT(mf).reference


def test_ghf_preserves_higher_precision_complex_initial_density():
    initial = np.array([[.5, -.5j], [.5j, .5]], dtype=np.complex128)
    result = scf.run_ghf_from_integrals(
        overlap=np.eye(1, dtype=np.float32), hcore=-np.eye(1, dtype=np.float32),
        eri=np.ones((1, 1, 1, 1), dtype=np.float32), nelectron=1,
        nuclear_repulsion=0., init_density=initial,
    )
    assert result.converged
    assert result.density_matrix.dtype == np.complex128
    np.testing.assert_allclose(result.density_matrix, initial, atol=1e-12, rtol=0)
