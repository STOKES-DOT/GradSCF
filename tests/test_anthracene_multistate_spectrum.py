"""Real spectrum example I/O with shifted two/eight-band VG fixtures.

The electronic input is synthetic. The already validated single-state helper
supplies independent local-axis area oracles; only bookkeeping is substituted.
"""
import csv
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType

import numpy as np
import pytest
from scipy.integrate import trapezoid

from gradscf_tools.vibronic import spectrum


SCRIPT = Path(__file__).resolve().parents[1]/'examples/tdscf/anthracene_ht/04_spectrum.py'
EV = 27.211386245988


def problem(tmp_path, monkeypatch, nstates=2, *, complete=True, matching_signature=True):
    output = tmp_path/'report'/'raw'
    output.mkdir(parents=True)
    omega = np.array([.007, .011])
    numbers = np.arange(nstates)
    energies = .12+.0173107*numbers
    gradients = np.column_stack((.00012*np.cos(numbers), -.00019*np.sin(numbers+.7)))
    dipoles = .45**numbers[:, None]*np.array([[.3, -.1, .02]])
    slopes = np.broadcast_to(np.array([[.004, -.003], [.001, .002], [-.001, .001]]),
                             (nstates, 3, 2))*(1+numbers[:, None, None]/5)
    np.savez(output/'mode_response.npz', frequencies=omega, gap_gradient=gradients,
             dipole=dipoles, dipole_derivative=slopes, vertical_energy=energies,
             signature='same-validated-response')
    (output/'mode_response_status.json').write_text(json.dumps(dict(
        complete=complete, signature='same-validated-response' if matching_signature else 'stale')))
    # Guard roots are real saved baseline inputs but must not enter the target
    # spectra or the displayed target vertical sticks.
    strengths = (2/3)*energies*np.sum(dipoles**2, axis=1)
    np.savez(output/'vertical.npz', excitation=np.r_[energies, [.37, .38, .39, .4]],
             oscillator_strength=np.r_[strengths, [100., 100., 100., 100.]])
    common = ModuleType('common')
    common.OUTPUT, common.BASE_OUTPUT = output, tmp_path/'base'/'raw'
    common.ROOT, common.SETTINGS = SCRIPT.parents[3], {'fwhm_cm1': 100.}

    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    common.save = save
    monkeypatch.setitem(sys.modules, 'common', common)
    return output, (omega, gradients, dipoles, slopes, energies)


@pytest.mark.parametrize('nstates', [2, 8])
def test_shifted_bands_share_photon_axis_and_preserve_absolute_optical_areas(
        tmp_path, monkeypatch, nstates):
    output, inputs = problem(tmp_path, monkeypatch, nstates)
    runpy.run_path(str(SCRIPT), run_name='__main__')
    omega, gradients, dipoles, slopes, energies = inputs
    with np.load(output/'spectrum.npz') as saved:
        axis = saved['energy_ev']
        total = saved['oscillator_density_per_ev']
        states = saved['state_oscillator_density_per_ev']
        assert states.shape == (nstates, axis.size, 4)
        assert total.shape == (axis.size, 4)
        np.testing.assert_array_equal(saved['target_roots'], np.arange(1, nstates+1))
        np.testing.assert_allclose(saved['vertical_energy_ev'], energies*EV, atol=1e-13)
        np.testing.assert_allclose(total, states.sum(axis=0), atol=2e-12)
        np.testing.assert_allclose(total[:, 3], total[:, :3].sum(axis=1), atol=2e-12)
        assert np.isfinite(states).all() and np.all(np.diff(axis) > 0)
        step = np.diff(axis)
        np.testing.assert_allclose(step, step[0], atol=1e-13, rtol=1e-10)
    expected = []
    for energy, gradient, dipole, slope in zip(energies, gradients, dipoles, slopes):
        local_axis, local_density, _ = spectrum(energy, omega, gradient, dipole, slope,
                                               fwhm_cm1=100., ntime=32768, time_step=10.)
        expected.append(trapezoid(local_density, local_axis, axis=0))
    expected = np.array(expected)
    integrated = trapezoid(states, axis, axis=1)
    scale = np.maximum(np.sum(abs(expected), axis=1), 1e-30)
    assert np.max(abs(integrated-expected)/scale[:, None]) < 1e-5
    np.testing.assert_allclose(trapezoid(total, axis, axis=0), integrated.sum(axis=0), atol=2e-12)
    assert integrated[0, 0] > 2*integrated[-1, 0], 'No per-state peak/area normalization'
    status = json.loads((output/'spectrum_status.json').read_text())
    assert status['complete'] and status['state_count'] == nstates
    assert status['target_roots'] == status['vertical_roots_plotted'] == list(range(1, nstates+1))
    assert max(status['checks']['interpolation_relative_area_error']) < 1e-5
    assert status['checks']['summed_relative_area_error'] < 1e-5
    with (output.parent/'modes.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == nstates*len(omega)
    with (output.parent/'bands.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == nstates
    assert len(status['figures']) >= 3
    for name in status['figures']:
        assert (output.parent/name).stat().st_size > 5000


def test_mode_response_signature_must_match_completed_status(tmp_path, monkeypatch):
    output, _ = problem(tmp_path, monkeypatch, matching_signature=False)
    with pytest.raises(RuntimeError, match='(?i)signature'):
        runpy.run_path(str(SCRIPT), run_name='__main__')
    assert not (output/'spectrum.npz').exists()


def test_incomplete_mode_responses_are_rejected_before_spectrum(tmp_path, monkeypatch):
    output, _ = problem(tmp_path, monkeypatch, complete=False)
    with pytest.raises(RuntimeError, match='(?i)(complete|validated)'):
        runpy.run_path(str(SCRIPT), run_name='__main__')
    assert not (output/'spectrum.npz').exists()
