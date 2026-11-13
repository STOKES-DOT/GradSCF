"""Independent 0-K same-curvature VG/linear-HT checks, atomic units.

Only the tiny oracle enumerates oscillator states. The production helper must
also accept all 66 molecular modes without a total-quanta state-space cutoff.
"""
import importlib
import importlib.util

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.special import factorial


CM1 = 219474.6313631965


def api():
    assert importlib.util.find_spec("gradscf_tools.vibronic") is not None, (
        "Missing reusable factorized VG FC/HT helper")
    return importlib.import_module("gradscf_tools.vibronic")


def inputs():
    return (np.array([.007, .011]), np.array([.0002, -.0003]),
            np.array([.3, -.15, .04]),
            np.array([[.005, -.004], [.002, .003], [-.001, .002]]))


def fock_reference(omega, gradient, dipole, slopes, count=14):
    annihilation = np.diag(np.sqrt(np.arange(1, count)), 1)
    operators = [np.kron(annihilation, np.eye(count)),
                 np.kron(np.eye(count), annihilation)]
    size = count**2
    vacuum = np.eye(size)[:, 0]
    hamiltonian = np.zeros((size, size))
    fc, ht = np.outer(vacuum, dipole), np.zeros((size, 3))
    for mode, operator in enumerate(operators):
        coordinate = (operator+operator.T)/np.sqrt(2*omega[mode])
        hamiltonian += omega[mode]*operator.T@operator+gradient[mode]*coordinate
        ht += np.outer(coordinate@vacuum, slopes[:, mode])
    energies, vectors = eigh(hamiltonian)
    fc, ht = vectors.T@fc, vectors.T@ht
    weights = np.column_stack((np.sum(fc*fc, axis=1), np.sum(ht*ht, axis=1),
                               2*np.sum(fc*ht, axis=1), np.sum((fc+ht)**2, axis=1)))
    return energies, weights


def test_all_four_correlations_match_independent_two_mode_fock_hamiltonian():
    module = api()
    omega, gradient, dipole, slopes = inputs()
    energies, weights = fock_reference(omega, gradient, dipole, slopes)
    times = np.linspace(-1500, 1500, 151)
    expected = np.exp(-1j*times[:, None]*energies)@weights
    actual = module.correlation(times, omega, gradient, dipole, slopes)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
    np.testing.assert_allclose(actual[:, 3], actual[:, 0]+actual[:, 1]+actual[:, 2], atol=2e-15)
    np.testing.assert_allclose(actual[::-1], actual.conj(), atol=2e-14)


def test_constant_dipole_matches_poisson_progression_at_large_displacement():
    module = api()
    omega, huang_rhys = .01, 5.3
    gradient = np.sqrt(2*omega**3*huang_rhys)
    dipole = np.array([.3, -.2, .1])
    times, numbers = np.linspace(-1200, 1200, 81), np.arange(75)
    probabilities = np.exp(-huang_rhys)*huang_rhys**numbers/factorial(numbers)
    expected = np.exp(-1j*times[:, None]*omega*(numbers-huang_rhys))@probabilities
    actual = module.correlation(times, [omega], [gradient], dipole, np.zeros((3, 1)))
    np.testing.assert_allclose(actual[:, 0], expected*np.dot(dipole, dipole), atol=2e-13)
    np.testing.assert_array_equal(actual[:, 1:3], 0.)
    np.testing.assert_array_equal(actual[:, 0], actual[:, 3])


def test_zero_displacement_dark_transition_has_finite_one_quantum_ht():
    module = api()
    omega, _, _, slopes = inputs()
    times = np.linspace(-200, 200, 21)
    expected = np.exp(-1j*times[:, None]*omega)@np.sum(slopes**2/(2*omega), axis=0)
    actual = module.correlation(times, omega, np.zeros(2), np.zeros(3), slopes)
    np.testing.assert_allclose(actual[:, 3], expected, atol=2e-15)
    np.testing.assert_array_equal(actual[:, [0, 2]], 0.)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("gauge", ["electronic_phase", "mode_gauge", "spatial_rotation"])
def test_signed_components_preserve_electronic_and_coordinate_gauges(gauge):
    module = api()
    omega, gradient, dipole, slopes = inputs()
    times = np.linspace(-500, 500, 51)
    expected = module.correlation(times, omega, gradient, dipole, slopes)
    if gauge == "electronic_phase":
        dipole, slopes = -dipole, -slopes
    elif gauge == "mode_gauge":
        order, signs = [1, 0], np.array([-1., 1.])
        omega, gradient, slopes = omega[order], gradient[order]*signs, slopes[:, order]*signs
    else:
        angle = .37
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        dipole, slopes = rotation@dipole, rotation@slopes
    np.testing.assert_allclose(module.correlation(times, omega, gradient, dipole, slopes),
                               expected, atol=2e-14)


def test_gaussian_broadening_keeps_absolute_strength_and_one_quantum_energy():
    module = api()
    omega, vertical, width = .01, .16, 100.
    dipole = np.array([.2, .1, -.05])
    slopes = np.array([[.004], [.002], [-.001]])
    axis, density, metadata = module.spectrum(vertical, [omega], [0.], dipole, slopes,
                                             fwhm_cm1=width, ntime=32768, time_step=25.)
    sigma = width/(CM1*2*np.sqrt(2*np.log(2)))
    def gaussian(center):
        return np.exp(-.5*((axis-center)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    expected = np.column_stack((np.dot(dipole, dipole)*gaussian(vertical),
        np.sum(slopes**2)/(2*omega)*gaussian(vertical+omega), np.zeros_like(axis)))
    expected = np.column_stack((expected, expected.sum(axis=1)))*(2/3)*axis[:, None]
    np.testing.assert_allclose(density, expected, atol=2e-11, rtol=2e-10)
    assert metadata["temperature_kelvin"] == 0.
    assert metadata["empirical_shift_ev"] == 0.
    assert metadata["columns"] == ["fc", "ht", "interference", "total"]


def test_displaced_fft_obeys_optical_moment_sum_rules_without_rescaling():
    module = api()
    omega, gradient, dipole, slopes = inputs()
    vertical = .16
    zero = np.array([np.dot(dipole, dipole), np.sum(slopes**2/(2*omega)), 0.])
    first = np.array([0., .5*np.sum(slopes**2), np.sum(gradient*(dipole@slopes)/omega)])
    expected = (2/3)*(vertical*zero+first)
    expected = np.r_[expected, expected.sum()]
    axis, density, metadata = module.spectrum(vertical, omega, gradient, dipole, slopes)
    np.testing.assert_allclose(np.sum(density, axis=0)*(axis[1]-axis[0]), expected,
                               atol=2e-12, rtol=2e-11)
    np.testing.assert_allclose(density[:, 3], density[:, 0]+density[:, 1]+density[:, 2], atol=2e-12)
    assert density[:, 2].min() < 0 < density[:, 2].max()
    assert density[:, [0, 1, 3]].min() >= -1e-10*max(density[:, 3])
    assert metadata["relative_strength_sum_rule_error"] < 1e-9


def test_sixty_six_modes_have_finite_factorized_closure_without_enumeration():
    module = api()
    omega = np.linspace(.002, .015, 66)
    gradient = np.linspace(-.00002, .00003, 66)
    slopes = np.linspace(-.0002, .0003, 198).reshape(3, 66)
    dipole = np.array([.2, .1, -.03])
    actual = module.correlation(np.array([-11., 0., 11.]), omega, gradient, dipole, slopes)
    closure = np.array([np.dot(dipole, dipole), np.sum(slopes**2/(2*omega)), 0.])
    np.testing.assert_allclose(actual[1].real, np.r_[closure, closure.sum()], atol=2e-14)
    assert actual.shape == (3, 4) and np.isfinite(actual).all()


@pytest.mark.parametrize("change", ["negative_frequency", "complex_dipole", "wrong_derivative", "nonfinite"])
def test_invalid_physical_inputs_are_rejected(change):
    module = api()
    omega, gradient, dipole, slopes = inputs()
    if change == "negative_frequency": omega[0] = -.01
    elif change == "complex_dipole": dipole = dipole.astype(complex)+1j
    elif change == "wrong_derivative": slopes = slopes.T
    else: gradient[0] = np.nan
    with pytest.raises(ValueError):
        module.correlation([0.], omega, gradient, dipole, slopes)


def test_insufficient_time_window_is_rejected_instead_of_negative_spectrum():
    module = api()
    with pytest.raises(RuntimeError, match="(?i)(window|sampling|resolution)"):
        module.spectrum(.16, [.01], [0.], [.3, 0., 0.], np.zeros((3, 1)),
                        ntime=256, time_step=1., fwhm_cm1=100.)
