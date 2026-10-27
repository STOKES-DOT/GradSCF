from pyscf_data_reference import true_df_factors_from_libcint_mol
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals import basis_from_pyscf_spec, evaluate_cartesian_ao
from gradscf.integrals import build_hcore, eri_pair_matrix_packed, eri_tensor, overlap_matrix
from gradscf.integrals.molecular.jk import build_j_from_df, build_jk_from_df, build_jk_from_df_orbitals
from gradscf.integrals.molecular.factorization import eri_to_df_factors
from gradscf.scf import RKSConfig, run_rks_from_integrals
from gradscf.scf.builders import _build_restricted_reference
from reference_scf_features import _restricted_response_eri_slices_from_mo_tensor
from gradscf.integrals.molecular.jk import build_jk_from_eri_pair_matrix
from gradscf.ao2mo import eri_pair_matrix_to_mo_eri_slices
from gradscf.tddft import RestrictedCasidaTDDFT
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional


def _pyscf_or_skip():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for density-fitting comparison tests.")


def _water_mol():
    from pyscf import gto

    return gto.M(
        atom="""
        O  0.000000  0.000000  0.117790
        H  0.000000  0.755453 -0.471161
        H  0.000000 -0.755453 -0.471161
        """,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        verbose=0,
    )


def _direct_provider(mol):
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    topology, parameters = integrals.prepare_basis(mol.atom, mol.basis, unit=mol.unit,
                                                   cart=True, spin=mol.spin, charge=mol.charge)
    return NativeDirectBasis(integrals.make_plan(topology), parameters)


def test_df_jk_matches_dense_eri_contractions_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    rng = np.random.default_rng(0)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    k_ref = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    factors = eri_to_df_factors(eri, tol=1e-10)
    j_df, k_df = build_jk_from_df(factors, density)

    assert np.allclose(np.asarray(j_df), j_ref, atol=2e-5, rtol=2e-5)
    assert np.allclose(np.asarray(k_df), k_ref, atol=2e-5, rtol=2e-5)


def test_df_j_only_matches_dense_coulomb_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    rng = np.random.default_rng(1)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    factors = eri_to_df_factors(eri, tol=1e-10)
    j_df = build_j_from_df(factors, density)

    assert np.allclose(np.asarray(j_df), j_ref, atol=2e-5, rtol=2e-5)


def test_df_orbital_jk_matches_density_jk_for_closed_shell_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    factors = eri_to_df_factors(eri, tol=1e-10)

    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(mol.nao_nr(), mol.nao_nr())))
    nocc = 3
    mo_occ = np.zeros((mol.nao_nr(),), dtype=float)
    mo_occ[:nocc] = 2.0
    density = q @ np.diag(mo_occ) @ q.T

    j_density, k_density = build_jk_from_df(factors, density)
    j_orb, k_orb = build_jk_from_df_orbitals(
        factors,
        density,
        q,
        mo_occ,
        nocc=nocc,
    )

    assert np.allclose(np.asarray(j_orb), np.asarray(j_density), atol=2e-5, rtol=2e-5)
    assert np.allclose(np.asarray(k_orb), np.asarray(k_density), atol=2e-5, rtol=2e-5)


def test_true_df_factors_from_libcint_mol_match_dense_jk_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    eri = np.asarray(mol.intor("int2e"), dtype=float)
    rng = np.random.default_rng(3)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    k_ref = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    factors = true_df_factors_from_libcint_mol(mol)
    j_df, k_df = build_jk_from_df(factors, density)

    assert np.allclose(np.asarray(j_df), j_ref, atol=3e-4, rtol=3e-4)
    assert np.allclose(np.asarray(k_df), k_ref, atol=3e-4, rtol=3e-4)


def test_packed_pair_matrix_matches_full_eri_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    pair = np.asarray(eri_pair_matrix_packed(basis), dtype=float)
    rows, cols = np.tril_indices(mol.nao_nr())
    pair_ref = eri[rows, cols][:, rows, cols]
    assert np.allclose(pair, pair_ref, atol=2e-8, rtol=2e-8)


def test_packed_no_df_jk_matches_dense_eri_contractions_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    pair = np.asarray(eri_pair_matrix_packed(basis), dtype=float)
    rng = np.random.default_rng(5)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    k_ref = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    j_pair, k_pair = build_jk_from_eri_pair_matrix(pair, density)

    assert np.allclose(np.asarray(j_pair), j_ref, atol=2e-5, rtol=2e-5)
    assert np.allclose(np.asarray(k_pair), k_ref, atol=2e-5, rtol=2e-5)


def test_direct_no_df_jk_matches_dense_eri_contractions_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    rng = np.random.default_rng(7)
    density = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = 0.5 * (density + density.T)

    j_ref = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    k_ref = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    direct_j, direct_k = _direct_provider(mol).get_jk(density, screening_threshold=0.0)

    assert np.allclose(np.asarray(direct_j), j_ref, atol=2e-5, rtol=2e-5)
    assert np.allclose(np.asarray(direct_k), k_ref, atol=2e-5, rtol=2e-5)








def test_rks_full_backend_accepts_packed_no_df_eri_for_water():
    _pyscf_or_skip()
    from pyscf import dft

    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    s = overlap_matrix(basis)
    h = build_hcore(basis)
    pair = eri_pair_matrix_packed(basis)

    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    coords = np.asarray(mf.grids.coords, dtype=float)
    weights = np.asarray(mf.grids.weights, dtype=float)
    ao_deriv1 = evaluate_cartesian_ao(basis, coords, deriv=1)
    ao = ao_deriv1[0]

    cfg = RKSConfig(
        xc_spec="pbe0",
        max_cycle=80,
        conv_tol=1e-9,
        conv_tol_density=1e-6,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="full",
    )
    out = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=None,
        eri_pair_matrix=pair,
        nelectron=mol.nelectron,
        nuclear_repulsion=float(mol.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=cfg,
    )

    assert out.converged
    assert np.isclose(out.total_energy, mf.e_tot, atol=2e-4, rtol=2e-6)


def test_rks_direct_backend_accepts_basis_without_prebuilt_eri_for_water():
    _pyscf_or_skip()
    from pyscf import dft

    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    s = overlap_matrix(basis)
    h = build_hcore(basis)

    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    coords = np.asarray(mf.grids.coords, dtype=float)
    weights = np.asarray(mf.grids.weights, dtype=float)
    ao_deriv1 = evaluate_cartesian_ao(basis, coords, deriv=1)
    ao = ao_deriv1[0]

    cfg = RKSConfig(
        xc_spec="pbe0",
        max_cycle=80,
        conv_tol=1e-9,
        conv_tol_density=1e-6,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="direct",
        direct_scf_tol=0.0,
    )
    out = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=None,
        direct_basis=_direct_provider(mol),
        nelectron=mol.nelectron,
        nuclear_repulsion=float(mol.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=cfg,
    )

    assert out.converged
    assert np.isclose(out.total_energy, mf.e_tot, atol=2e-4, rtol=2e-6)


def test_rks_direct_backend_matches_pyscf_for_water():
    _pyscf_or_skip()
    from pyscf import dft

    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    s = overlap_matrix(basis)
    h = build_hcore(basis)

    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    coords = np.asarray(mf.grids.coords, dtype=float)
    weights = np.asarray(mf.grids.weights, dtype=float)
    ao_deriv1 = evaluate_cartesian_ao(basis, coords, deriv=1)
    ao = ao_deriv1[0]

    cfg = RKSConfig(
        xc_spec="pbe0",
        max_cycle=80,
        conv_tol=1e-9,
        conv_tol_density=1e-6,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="direct",
        direct_scf_tol=0.0,
    )
    out = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=None,
        direct_basis=_direct_provider(mol),
        nelectron=mol.nelectron,
        nuclear_repulsion=float(mol.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=cfg,
    )

    assert out.converged
    assert np.isclose(out.total_energy, mf.e_tot, atol=2e-4, rtol=2e-6)


def test_packed_no_df_mo_slices_match_dense_eri_for_water():
    _pyscf_or_skip()
    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=3,
    )
    eri = np.asarray(eri_tensor(basis), dtype=float)
    pair = np.asarray(eri_pair_matrix_packed(basis), dtype=float)
    rng = np.random.default_rng(6)
    q, _ = np.linalg.qr(rng.normal(size=(mol.nao_nr(), mol.nao_nr())))
    nocc = 3

    dense_slices = _restricted_response_eri_slices_from_mo_tensor(
        eri,
        q,
        nocc,
        include_oovv=True,
    )
    packed_slices = eri_pair_matrix_to_mo_eri_slices(
        pair,
        q,
        nocc=nocc,
        include_oovv=True,
    )

    for packed, dense in zip(packed_slices, dense_slices, strict=True):
        assert packed is not None
        assert dense is not None
        assert np.allclose(np.asarray(packed), np.asarray(dense), atol=2e-5, rtol=2e-5)


def test_rks_df_backend_matches_pyscf_water_total_energy():
    _pyscf_or_skip()
    from pyscf import dft

    mol = _water_mol()
    basis = basis_from_pyscf_spec(
        mol.atom,
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        spin=0,
        charge=0,
        max_l=1,
    )
    s = overlap_matrix(basis)
    h = build_hcore(basis)
    eri = eri_tensor(basis)

    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    coords = np.asarray(mf.grids.coords, dtype=float)
    weights = np.asarray(mf.grids.weights, dtype=float)
    ao_deriv1 = evaluate_cartesian_ao(basis, coords, deriv=1)
    ao = ao_deriv1[0]

    cfg = RKSConfig(
        xc_spec="pbe",
        max_cycle=80,
        conv_tol=1e-9,
        conv_tol_density=1e-6,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="df",
        df_tol=1e-10,
    )
    out = run_rks_from_integrals(
        overlap=s,
        hcore=h,
        eri=eri,
        nelectron=mol.nelectron,
        nuclear_repulsion=float(mol.energy_nuc()),
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=weights,
        config=cfg,
    )

    assert out.converged
    assert np.isclose(out.total_energy, mf.e_tot, atol=5e-5, rtol=5e-5)


def test_native_spectral_df_reference_for_water_packs_factors():
    _pyscf_or_skip()
    from pyscf import dft

    mol = _water_mol()
    mf = dft.RKS(mol)
    mf.xc = "hf"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    cfg = RKSConfig(
        xc_spec="hf",
        max_cycle=50,
        conv_tol=1e-9,
        conv_tol_density=1e-7,
        damping=0.15,
        potential_clip=20.0,
        jk_backend="df",
        df_tol=1e-10,
    )
    ref = _build_restricted_reference(
        atom=mol.atom,
        basis="sto-3g",
        xc_spec="hf",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=cfg,
        grid_ao_backend="jax",
    )

    assert ref.df_factors is not None
    assert ref.eri_ovov is None
    assert ref.eri_ovvo is None
    assert ref.eri_oovv is None
    assert np.asarray(ref.rep_tensor).size == 0
    assert np.isclose(ref.mf_energy, mf.e_tot, atol=5e-5, rtol=5e-5)


def test_native_spectral_df_reference_matches_full_eri_reference():
    _pyscf_or_skip()
    from pyscf import dft

    import gradscf.scf.builders as reference_mod

    mol = _water_mol()
    mf = dft.RKS(mol)
    mf.xc = "hf"
    mf.grids.level = 0
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.kernel()
    assert mf.converged

    ref = reference_mod._build_restricted_reference(
        atom=mol.atom,
        basis="sto-3g",
        xc_spec="hf",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="hf",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            potential_clip=20.0,
            jk_backend="df",
            df_tol=1e-10,
        ),
        grid_ao_backend="jax",
    )

    assert ref.df_factors is not None
    assert ref.eri_ovov is None
    assert ref.eri_ovvo is None
    assert np.asarray(ref.rep_tensor).size == 0
    assert np.isclose(ref.mf_energy, mf.e_tot, atol=2e-4, rtol=2e-4)


def test_native_spectral_df_reference_preserves_backend_when_xc_overrides_config():
    _pyscf_or_skip()

    import gradscf.scf.builders as reference_mod

    ref = reference_mod._build_restricted_reference(
        atom=_water_mol().atom,
        basis="sto-3g",
        xc_spec="hf",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            potential_clip=20.0,
            jk_backend="df",
            df_tol=1e-10,
        ),
        grid_ao_backend="jax",
    )

    assert ref.df_factors is not None
    assert np.asarray(ref.rep_tensor).size == 0


def test_df_reference_lazy_slices_support_tda():
    _pyscf_or_skip()
    ref = _build_restricted_reference(
        atom=_water_mol().atom,
        basis="sto-3g",
        xc_spec="pbe0",
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        grids_level=0,
        max_l=1,
        rks_config=RKSConfig(
            xc_spec="pbe0",
            max_cycle=50,
            conv_tol=1e-9,
            conv_tol_density=1e-7,
            damping=0.15,
            potential_clip=20.0,
            jk_backend="df",
            df_tol=1e-10,
        ),
        grid_ao_backend="jax",
    )

    assert ref.df_factors is not None
    assert ref.eri_ovov is None
    solver = RestrictedCasidaTDDFT(
        molecule=ref,
        xc_functional=SemilocalResponseFunctional("pbe0"),
    )
    result = solver.tda(nstates=1)
    assert np.asarray(result.excitation_energies).shape == (1,)
    assert np.isfinite(np.asarray(result.excitation_energies)).all()
