"""Tests for the imaginary-frequency quadrature grid."""

import numpy as np

from gradscf.gw import scaled_legendre_grid


def test_scaled_legendre_grid_matches_reference_mapping():
    nw = 32
    freqs, wts = scaled_legendre_grid(nw)
    x, w = np.polynomial.legendre.leggauss(nw)
    x0 = 0.5
    np.testing.assert_allclose(freqs, x0 * (1.0 + x) / (1.0 - x), rtol=1e-14)
    np.testing.assert_allclose(wts, w * 2.0 * x0 / (1.0 - x) ** 2, rtol=1e-14)
    assert freqs.shape == (nw,)
    assert np.all(np.asarray(freqs) > 0.0)
    assert np.all(np.asarray(wts) > 0.0)


def test_scaled_legendre_grid_integrates_constant_shift():
    # ∫_0^inf dx0-mapped weight should approximate ∫ 1/(1+w^2) dw = pi/2
    # for a smooth decaying integrand once nw is moderately large.
    nw = 200
    freqs, wts = scaled_legendre_grid(nw)
    val = float(np.sum(np.asarray(wts) / (1.0 + np.asarray(freqs) ** 2)))
    assert abs(val - np.pi / 2.0) < 1e-6
