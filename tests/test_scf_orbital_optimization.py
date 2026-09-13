"""Opt-in fixed-occupation minimization preserves the orbital manifold."""
import numpy as np
import pytest

from gradscf.scf.orbital_optimization import minimize_uks_from_integrals, minimize_roks_from_integrals


def _two_orbital_inputs():
    angle = 0.37
    c = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return dict(overlap=np.eye(2), hcore=np.diag([-1., .5]), eri=np.zeros((2, 2, 2, 2)),
                nuclear_repulsion=0., ao=np.zeros((0, 2)), ao_deriv1=np.zeros((4, 0, 2)),
                grid_weights=np.zeros(0)), c


def test_unrestricted_orbital_minimization_lowers_energy_and_preserves_electrons():
    args, c = _two_orbital_inputs()
    result = minimize_uks_from_integrals(
        **args, mo_coeff=np.stack([c, c]), mo_occ=np.array([[1., 0.], [0., 0.]]), xc_spec="hf",
    )
    assert result.stationary
    assert result.gradient_norm < 1e-7
    np.testing.assert_allclose(result.total_energy, -1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.trace(result.density_matrix, axis1=1, axis2=2), [1., 0.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.mo_coeff.transpose(0, 2, 1) @ result.mo_coeff,
                               np.broadcast_to(np.eye(2), (2, 2, 2)), atol=1e-12, rtol=0)


def test_orbital_minimization_does_not_aufbau_refill_stationary_excited_state():
    args, _ = _two_orbital_inputs()
    result = minimize_uks_from_integrals(
        **args, mo_coeff=np.stack([np.eye(2), np.eye(2)]),
        mo_occ=np.array([[0., 1.], [0., 0.]]), xc_spec="hf",
    )
    assert result.stationary  # Stationarity is deliberately not a stability claim.
    np.testing.assert_allclose(result.total_energy, .5, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result.mo_occ, [[0., 1.], [0., 0.]])


def test_rohf_orbital_minimization_matches_reference_and_keeps_common_orbitals():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from scipy.linalg import eigh
    mol = gto.M(atom="Li 0 0 0;H 0 0 1.6", basis="sto-3g", charge=1, spin=1, verbose=0)
    ref = scf.ROHF(mol).run(conv_tol=1e-12)
    s, h = ref.get_ovlp(), ref.get_hcore()
    _, c = eigh(h, s)
    occ = np.zeros((2, len(c))); occ[0, :2] = 1.; occ[1, :1] = 1.
    result = minimize_roks_from_integrals(
        overlap=s, hcore=h, eri=mol.intor("int2e"), nuclear_repulsion=mol.energy_nuc(),
        ao=np.zeros((0, len(c))), ao_deriv1=np.zeros((4, 0, len(c))), grid_weights=np.zeros(0),
        mo_coeff=c, mo_occ=occ, xc_spec="hf", max_iterations=300,
    )
    assert result.stationary
    np.testing.assert_allclose(result.total_energy, ref.e_tot, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.mo_coeff.T @ s @ result.mo_coeff, np.eye(len(c)), atol=1e-10, rtol=0)
    a, b = result.density_matrix
    np.testing.assert_allclose(a @ s @ b, b, atol=1e-10, rtol=0)


def test_orbital_minimization_rejects_nonorthonormal_initial_orbitals():
    args, c = _two_orbital_inputs()
    with pytest.raises(ValueError, match="orthonormal"):
        minimize_uks_from_integrals(**args, mo_coeff=np.stack([2*c, c]),
            mo_occ=np.array([[1., 0.], [0., 0.]]), xc_spec="hf")


def test_generalized_complex_orbital_minimization_preserves_spinor_manifold():
    from scipy.linalg import expm
    from gradscf.scf.orbital_optimization import minimize_gks_from_integrals
    args, _ = _two_orbital_inputs()
    generator = np.zeros((4, 4), complex)
    generator[0, 3] = .3j; generator[3, 0] = .3j
    result = minimize_gks_from_integrals(
        **args, mo_coeff=expm(generator), mo_occ=np.array([1., 0., 0., 0.]), xc_spec="hf",
    )
    assert result.stationary
    np.testing.assert_allclose(result.total_energy, -1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.mo_coeff.conj().T @ result.mo_coeff, np.eye(4), atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.density_matrix @ result.density_matrix,
                               result.density_matrix, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.trace(result.density_matrix), 1., atol=1e-12, rtol=0)


def test_orbital_minimizer_checks_stationarity_below_total_energy_resolution():
    args, _ = _two_orbital_inputs()
    args['nuclear_repulsion'] = 1e6
    angle = 1e-6
    c = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result = minimize_uks_from_integrals(
        **args, mo_coeff=np.stack([c, c]), mo_occ=np.array([[1., 0.], [0., 0.]]),
        xc_spec="hf", gradient_tolerance=1e-9,
    )
    assert result.stationary
    assert result.gradient_norm < 1e-9
    # The JAX line search can now resolve this case without a Newton correction.
    np.testing.assert_allclose(result.total_energy, 999999., atol=1e-9, rtol=0)


def test_orbital_polishing_handles_weak_negative_curvature_without_ascent():
    # A nearly degenerate occupied/virtual pair has weak negative curvature,
    # while a separate stiff direction still needs stationarity polishing.
    # Unshifted Newton heads toward the saddle and is not a descent direction.
    from scipy.linalg import expm
    generator = np.zeros((3, 3))
    generator[0, 1], generator[1, 0] = .03, -.03
    generator[0, 2], generator[2, 0] = 1e-6, -1e-6
    coeff = expm(generator)
    result = minimize_uks_from_integrals(
        overlap=np.eye(3), hcore=np.diag([0., -1e-8, 1.]),
        eri=np.zeros((3, 3, 3, 3)), nuclear_repulsion=1e6,
        ao=np.zeros((0, 3)), ao_deriv1=np.zeros((4, 0, 3)), grid_weights=np.zeros(0),
        mo_coeff=np.stack([coeff, coeff]), mo_occ=np.array([[1., 0., 0.], [0., 0., 0.]]),
        xc_spec="hf", gradient_tolerance=1e-9,
    )
    assert result.stationary
    assert result.gradient_norm < 1e-9
    accepted = [step for step in result.polishing_history if step['accepted']]
    assert all(step['accepted_step_norm'] <= .1 for step in accepted)
    assert all(step['gradient_after'] < step['gradient_before']
               if not step['refinement'] else step['gradient_after'] <= 1e-9
               for step in accepted)
    assert all(step['energy_change'] <= step['energy_allowance'] for step in accepted)
