"""Independent comparison optimizer: Frechet derivatives and physical residuals."""
import numpy as np
import pytest
from scipy.linalg import eigh, expm

pytest.importorskip("pyscf")

from comparisons.pyscf_orbital_minimizer import OrbitalObjective, minimize_pyscf_orbitals


@pytest.mark.parametrize('method', ['UHF', 'ROHF', 'GHF'])
def test_reference_frechet_gradient_agrees_with_finite_difference(method):
    from pyscf import gto, scf
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', charge=1, spin=1, verbose=0)
    mf = getattr(scf, method)(mol)
    _, c = eigh(mol.intor('int1e_kin') + mol.intor('int1e_nuc'), mol.intor('int1e_ovlp'))
    occ = np.zeros((2, len(c))); occ[0, :2] = 1; occ[1, :1] = 1
    if method == 'UHF': c = np.stack([c, c])
    if method == 'GHF': c = np.kron(np.eye(2), c).astype(complex); occ = occ.ravel()
    objective = OrbitalObjective(mf, c, occ, method=method)
    rng = np.random.default_rng(903)
    x = rng.normal(size=objective.size) * .1
    _, grad = objective.value_and_grad(x)
    # Nonzero rotations exercise the Frechet adjoint, including imaginary angles.
    for direction in rng.normal(size=(5, objective.size)):
        direction /= np.linalg.norm(direction)
        step = 1e-5
        fd = (objective.value_and_grad(x + step*direction)[0] -
              objective.value_and_grad(x - step*direction)[0])/(2*step)
        np.testing.assert_allclose(grad @ direction, fd, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize('method', ['UHF', 'ROHF', 'GHF'])
def test_reference_orbital_minimizer_matches_lih_plus_hf(method):
    from pyscf import gto, scf
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', charge=1, spin=1, verbose=0)
    mf = getattr(scf, method)(mol)
    ref = mf.copy().run(conv_tol=1e-12, conv_tol_grad=1e-8)
    s = mol.intor('int1e_ovlp')
    _, c = eigh(mol.intor('int1e_kin') + mol.intor('int1e_nuc'), s)
    occ = np.zeros((2, len(c))); occ[0, :2] = 1; occ[1, :1] = 1
    if method == 'UHF': c = np.stack([c, c])
    if method == 'GHF': c = np.kron(np.eye(2), c).astype(complex); occ = occ.ravel(); s = np.kron(np.eye(2), s)
    result = minimize_pyscf_orbitals(mf, c, occ, method=method)
    assert result.stationary
    assert result.gradient_norm <= 1e-7
    np.testing.assert_allclose(result.total_energy, ref.e_tot, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.mo_coeff.conj().swapaxes(-1, -2) @ s @ result.mo_coeff,
                               np.broadcast_to(np.eye(c.shape[-1]), c.shape), atol=1e-10, rtol=0)
    np.testing.assert_array_equal(result.mo_occ, occ)
    final = OrbitalObjective(mf, result.mo_coeff, occ, method=method)
    true_norm = np.linalg.norm(final.value_and_grad(np.zeros(final.size))[1])/2
    np.testing.assert_allclose(result.gradient_norm, true_norm, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize('method', ['UKS', 'ROKS', 'GKS_ncol'])
def test_reference_dft_gradient_agrees_with_finite_difference(method):
    from pyscf import gto, dft
    mol = gto.M(atom='O 0 0 0; H 0 0 .97', basis='sto-3g', spin=1, verbose=0)
    mf = getattr(dft, method.replace('_ncol', ''))(mol)
    mf.xc = 'LDA_X + LDA_C_VWN'; mf.grids.level = 0; mf.small_rho_cutoff = 0.
    if method == 'GKS_ncol': mf.collinear = 'ncol'
    _, c = eigh(mol.intor('int1e_kin') + mol.intor('int1e_nuc'), mol.intor('int1e_ovlp'))
    occ = np.zeros((2, len(c))); occ[0, :5] = 1; occ[1, :4] = 1
    if method == 'UKS': c = np.stack([c, c])
    if method == 'GKS_ncol': c = np.kron(np.eye(2), c).astype(complex); occ = occ.ravel()
    objective = OrbitalObjective(mf, c, occ, method=method)
    rng = np.random.default_rng(901)
    x = rng.normal(size=objective.size) * .1
    _, grad = objective.value_and_grad(x)
    for direction in rng.normal(size=(3, objective.size)):
        direction /= np.linalg.norm(direction)
        step = 1e-5
        fd = (objective.value_and_grad(x + step*direction)[0] -
              objective.value_and_grad(x - step*direction)[0])/(2*step)
        np.testing.assert_allclose(grad @ direction, fd, atol=2e-7, rtol=2e-6)


def test_reference_polish_uses_true_gradient_below_energy_resolution():
    from pyscf import gto, scf
    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', charge=1, spin=1, verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-13)
    coefficients = mf.mo_coeff.copy()
    k = np.array([[0., 1e-6], [-1e-6, 0.]])
    coefficients[0] = coefficients[0] @ expm(k)
    # An additive constant leaves physical Fock and stationarity unchanged.
    mf.energy_nuc = lambda: 1e6
    result = minimize_pyscf_orbitals(mf, coefficients, mf.mo_occ, method='UHF',
                                   gradient_tolerance=1e-10)
    assert result.stationary
    assert result.gradient_norm <= 1e-10
    assert result.polish_steps > 0
    for step in result.polish_history:
        assert step['gradient_after'] < step['gradient_before']
        assert step['energy_change'] <= step['energy_allowance']
        assert step['step_norm'] <= .05


def test_reference_keeps_fixed_occupation_stationary_excited_state():
    from pyscf import gto, scf
    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', charge=1, spin=1, verbose=0)
    mf = scf.UHF(mol)
    eps, c = eigh(mf.get_hcore(), mf.get_ovlp())
    occ = np.array([[0., 1.], [0., 0.]])
    result = minimize_pyscf_orbitals(mf, np.stack([c, c]), occ, method='UHF')
    assert result.stationary  # A first-order condition does not imply the ground state.
    np.testing.assert_array_equal(result.mo_occ, occ)
    np.testing.assert_allclose(result.total_energy, eps[1] + mol.energy_nuc(), atol=1e-12)


def test_reference_budget_exhaustion_does_not_claim_stationarity():
    from pyscf import gto, scf
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', charge=1, spin=1, verbose=0)
    mf = scf.ROHF(mol)
    _, c = eigh(mf.get_hcore(), mf.get_ovlp())
    occ = np.zeros((2, len(c))); occ[0, :2] = 1; occ[1, :1] = 1
    result = minimize_pyscf_orbitals(mf, c, occ, method='ROHF', max_iterations=1)
    assert not result.optimizer_success
    assert not result.stationary
    assert result.gradient_norm > 1e-7
