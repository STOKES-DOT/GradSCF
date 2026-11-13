"""Factorized 0-K FC/linear-HT absorption in the vertical-gradient model.

One isolated real electronic state has H_e=E_vert+sum(omega*b^dagger*b+g*Q),
Q=(b+b^dagger)/sqrt(2*omega), and mu(Q)=mu0+kappa@Q. Both surfaces share
one harmonic curvature: no excited-state frequency change, Duschinsky mixing,
electronic-state coupling or thermal initial population is included.

Inputs use atomic units: omega/E in Hartree, Q in bohr*sqrt(electron mass),
g=d(E_e-E_g)/dQ, mu in e*bohr and kappa=dmu/dQ. This host-side postprocessing
consumes electronic derivatives; it does not supply their SCF/TDDFT response.
The analytic oscillator product retains every mode without Fock enumeration.
"""
import numpy as np
from scipy.constants import physical_constants


HARTREE_TO_WAVENUMBER = physical_constants["hartree-inverse meter relationship"][0]/100
COLUMNS = ["fc", "ht", "interference", "total"]


def _inputs(time, frequencies, gap_gradient, dipole, dipole_derivative):
    arrays = tuple(np.asarray(x) for x in
                   (time, frequencies, gap_gradient, dipole, dipole_derivative))
    time, omega, gradient, mu, slopes = arrays
    if (time.ndim != 1 or omega.ndim != 1 or not omega.size
            or gradient.shape != omega.shape or mu.shape != (3,)
            or slopes.shape != (3, omega.size)):
        raise ValueError("Inconsistent time, normal-mode or optical dimensions.")
    if (any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in arrays)
            or np.any(omega <= 0)):
        raise ValueError("Use finite real inputs and strictly positive mode energies.")
    return tuple(x.astype(np.float64, copy=False) for x in arrays)


def correlation(time, frequencies, gap_gradient, dipole, dipole_derivative):
    """Return FC, HT, signed interference and total correlations, shape (nt,4).

    The fast phase exp(-i*E_vert*t) is omitted; time is in inverse Hartree.
    C(t) sums signed transition amplitudes before squaring. Its zero-time
    total is |mu0|^2+sum(kappa^2/(2*omega)); no component is renormalized.
    The reorganization phase retains the model origin E00=E_vert-sum(g^2/2w^2).
    Zero displacements need no division by a Huang-Rhys amplitude.
    """
    time, omega, gradient, mu, slopes = _inputs(
        time, frequencies, gap_gradient, dipole, dipole_derivative)
    alpha = gradient/np.sqrt(2*omega**3)
    huang_rhys = alpha**2
    reorganization = np.dot(huang_rhys, omega)
    phase = np.exp(-1j*time[:, None]*omega)
    vacuum = np.exp(1j*reorganization*time+(phase-1)@huang_rhys)
    one_quantum = slopes/np.sqrt(2*omega)
    mixed = ((phase-1)*alpha)@one_quantum.T
    fc = vacuum*np.dot(mu, mu)
    ht = vacuum*(np.sum(mixed*mixed, axis=1)+phase@np.sum(one_quantum**2, axis=0))
    interference = 2*vacuum*(mixed@mu)
    result = np.column_stack((fc, ht, interference, fc+ht+interference))
    if not np.isfinite(result).all():
        raise FloatingPointError("Factorized oscillator correlation is nonfinite.")
    return result


def spectrum(vertical_energy, frequencies, gap_gradient, dipole, dipole_derivative,
             *, fwhm_cm1=100., ntime=32768, time_step=25.):
    """Return photon axis, four optical densities per Hartree and diagnostics.

    A unit-area Gaussian broadens the dipole spectral density, then multiplication
    by (2/3)*photon_energy gives the oscillator-strength density. This convention
    differs from first weighting discrete lines and then broadening their f.
    Interference remains signed; tiny FFT roundoff is preserved without clipping.
    No empirical energy shift or component-wise intensity rescaling is applied.
    Inadequate time windows or aliased optical sum rules are rejected.
    """
    scalars = np.asarray((vertical_energy, fwhm_cm1, time_step))
    if (np.iscomplexobj(scalars) or not np.isfinite(scalars).all()
            or np.any(scalars <= 0) or isinstance(ntime, (bool, np.bool_))
            or not isinstance(ntime, (int, np.integer)) or ntime < 256 or ntime % 2):
        raise ValueError("Use positive finite energy/width/step and an even ntime >= 256.")
    vertical_energy, fwhm_cm1, time_step = map(float, scalars)
    times = (np.arange(ntime)-ntime//2)*time_step
    times, omega, gradient, mu, slopes = _inputs(
        times, frequencies, gap_gradient, dipole, dipole_derivative)
    huang_rhys = gradient**2/(2*omega**3)
    reorganization = float(huang_rhys@omega)
    if vertical_energy <= reorganization:
        raise ValueError("The model 0-0 photon energy must be positive.")
    sigma = fwhm_cm1/(HARTREE_TO_WAVENUMBER*2*np.sqrt(2*np.log(2)))
    window = np.exp(-.5*(sigma*times)**2)
    if window[0] > 1e-10:
        raise RuntimeError("Time window is too short for the requested Gaussian resolution.")
    values = correlation(times, omega, gradient, mu, slopes)*window[:, None]
    transformed = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(values, axes=0), axis=0), axes=0)
    transformed *= ntime*time_step/(2*np.pi)
    axis = vertical_energy+np.fft.fftshift(2*np.pi*np.fft.fftfreq(ntime, time_step))
    imaginary = float(np.max(np.abs(transformed.imag)))
    real = transformed.real
    maximum = max(float(np.max(np.abs(real[:, [0, 1, 3]]))), 1e-30)
    minimum = float(np.min(real[:, [0, 1, 3]]))
    if imaginary > 1e-8*maximum or minimum < -1e-8*maximum:
        raise RuntimeError("Time-window or sampling artifacts exceed roundoff tolerance.")
    density = (2/3)*axis[:, None]*real
    zero = np.array([np.dot(mu, mu), np.sum(slopes**2/(2*omega)), 0.])
    first = np.array([0., .5*np.sum(slopes**2), np.sum(gradient*(mu@slopes)/omega)])
    expected = (2/3)*(vertical_energy*zero+first)
    expected = np.r_[expected, expected.sum()]
    integrated = np.sum(density, axis=0)*(2*np.pi/(ntime*time_step))
    scale = max(float(np.sum(np.abs(expected))), 1e-30)
    error = float(np.max(np.abs(integrated-expected))/scale)
    if error > 1e-8:
        raise RuntimeError("Sampling energy window aliases the optical moment sum rules.")
    metadata = dict(
        model="0-K same-curvature vertical-gradient harmonic model, linear Herzberg-Teller",
        columns=COLUMNS.copy(), energy_unit="Hartree", density_unit="oscillator strength per Hartree",
        broadening="unit-area Gaussian dipole density, then photon-energy weighting",
        fwhm_cm1=fwhm_cm1, temperature_kelvin=0., empirical_shift_ev=0.,
        ntime=int(ntime), time_step_au=time_step, mode_count=int(omega.size),
        zero_zero_energy_hartree=vertical_energy-reorganization,
        reorganization_energy_hartree=reorganization, huang_rhys=huang_rhys.tolist(),
        maximum_imaginary_density=imaginary, minimum_nonnegative_dipole_density=minimum,
        expected_optical_strengths=expected.tolist(), integrated_optical_strengths=integrated.tolist(),
        relative_strength_sum_rule_error=error)
    return axis, density, metadata


__all__ = ["correlation", "spectrum"]
