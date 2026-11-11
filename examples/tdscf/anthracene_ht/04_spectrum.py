"""Independent singlet VG/linear-HT bands on one physical photon-energy axis.

Each isolated electronic state retains all modes and its own FC/HT interference.
The electronic-state spectra are summed as intensities, without cross-state
coherence or separate peak normalization. Gaussian broadening is a display
convolution, not a lifetime or finite-temperature model.
"""
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from common import OUTPUT, SETTINGS, save
from gradscf_tools.vibronic import spectrum

HARTREE_TO_EV = 27.211386245988
CM1 = 219474.6313631965
FIGURES = OUTPUT.parent
COLUMNS = ['fc', 'ht', 'interference', 'total']
AREA_TOLERANCE = 1e-5

status = json.loads((OUTPUT/'mode_response_status.json').read_text())
if not status.get('complete'):
    raise RuntimeError('Validated complete normal-mode derivatives are required.')
with np.load(OUTPUT/'mode_response.npz') as stored:
    data = {name: stored[name].copy() for name in stored.files}
signature = str(data['signature'])
if status.get('signature') != signature:
    raise RuntimeError('The mode-response signature differs from its completed status.')
# Retain the original single-band file convention as well as the state axis.
vertical_energy = np.atleast_1d(data['vertical_energy'])
frequencies = data['frequencies']
gradients = data['gap_gradient'][None] if data['gap_gradient'].ndim == 1 else data['gap_gradient']
dipoles = data['dipole'][None] if data['dipole'].ndim == 1 else data['dipole']
slopes = data['dipole_derivative'][None] if data['dipole_derivative'].ndim == 2 else data['dipole_derivative']
nstates, nmodes = len(vertical_energy), len(frequencies)
if (vertical_energy.ndim != 1 or not nstates or frequencies.ndim != 1
        or gradients.shape != (nstates, nmodes) or dipoles.shape != (nstates, 3)
        or slopes.shape != (nstates, 3, nmodes)):
    raise RuntimeError('Complete state-by-mode response arrays are required.')
with np.load(OUTPUT/'vertical.npz') as baseline:
    np.testing.assert_allclose(baseline['excitation'][:nstates], vertical_energy, atol=1e-9, rtol=0)
    vertical_strengths = baseline['oscillator_strength'][:nstates].copy()
roots = np.arange(1, nstates+1)
save('spectrum_status.json', dict(complete=False, stage='assembling', signature=signature))

local = [spectrum(float(energy), frequencies, gradient, dipole, derivative,
                  fwhm_cm1=SETTINGS['fwhm_cm1'], ntime=32768, time_step=10.)
         for energy, gradient, dipole, derivative in zip(vertical_energy, gradients, dipoles, slopes)]
step = float(local[0][0][1]-local[0][0][0])
for axis, _, _ in local:
    np.testing.assert_allclose(np.diff(axis), step, atol=1e-14, rtol=1e-10)
left = min(axis[0] for axis, _, _ in local)
right = max(axis[-1] for axis, _, _ in local)
# Equal-spacing linear resampling with zero-padded endpoints preserves area;
# the complete FFT intervals are retained, including their roundoff tails.
axis = left-step+np.arange(int(np.ceil((right-left)/step))+3)*step
states = np.array([np.column_stack([np.interp(axis, original, density[:, column], left=0., right=0.)
                   for column in range(4)]) for original, density, _ in local])
density = states.sum(axis=0)
original_areas = np.array([trapezoid(values, original, axis=0) for original, values, _ in local])
state_areas = trapezoid(states, axis, axis=1)
area_scale = np.maximum(np.sum(np.abs(original_areas), axis=1), 1e-30)
area_errors = np.max(np.abs(state_areas-original_areas), axis=1)/area_scale
summed_area = trapezoid(density, axis, axis=0)
expected_area = np.sum([checks['expected_optical_strengths'] for _, _, checks in local], axis=0)
summed_error = float(np.max(np.abs(summed_area-expected_area))/max(np.sum(abs(expected_area)), 1e-30))
if np.max(area_errors) > AREA_TOLERANCE or summed_error > AREA_TOLERANCE:
    raise RuntimeError('Common-axis resampling does not preserve absolute optical strength.')
np.testing.assert_allclose(density[:, 3], density[:, :3].sum(axis=1), atol=2e-12, rtol=2e-12)

energy, per_ev, per_state = axis*HARTREE_TO_EV, density/HARTREE_TO_EV, states/HARTREE_TO_EV
checks = [item[2] for item in local]
origins = np.array([item['zero_zero_energy_hartree'] for item in checks])*HARTREE_TO_EV
hr = np.array([item['huang_rhys'] for item in checks])
np.savez_compressed(OUTPUT/'spectrum.npz', energy_ev=energy, oscillator_density_per_ev=per_ev,
    state_oscillator_density_per_ev=per_state, target_roots=roots, columns=COLUMNS,
    vertical_energy_ev=vertical_energy*HARTREE_TO_EV, zero_zero_energy_ev=origins, signature=signature)
np.savetxt(FIGURES/'spectrum.csv', np.column_stack((energy, per_ev)), delimiter=',',
    header='energy_eV,FC_per_eV,HT_per_eV,interference_per_eV,total_per_eV', comments='')
with (FIGURES/'bands.csv').open('w') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['state', 'vertical_eV', 'model_0_0_eV', 'reorganization_eV', 'sum_Huang_Rhys',
                     'integrated_FC', 'integrated_HT', 'integrated_interference', 'integrated_total',
                     'resampling_relative_area_error'])
    for index in range(nstates):
        writer.writerow([roots[index], vertical_energy[index]*HARTREE_TO_EV, origins[index],
            checks[index]['reorganization_energy_hartree']*HARTREE_TO_EV, hr[index].sum(),
            *state_areas[index], area_errors[index]])
with (FIGURES/'modes.csv').open('w') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['state', 'mode', 'frequency_cm-1', 'Huang_Rhys', 'd_gap_dQ_me',
                     'd_mu_x_dQ_me', 'd_mu_y_dQ_me', 'd_mu_z_dQ_me'])
    for state in range(nstates):
        for mode in range(nmodes):
            writer.writerow([roots[state], mode+1, frequencies[mode]*CM1, hr[state, mode],
                             gradients[state, mode], *slopes[state, :, mode]])

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
figures = []


def save_figure(figure, name):
    for suffix in ('png', 'pdf'):
        figure.savefig(FIGURES/f'{name}.{suffix}', dpi=220)
    figures.append(f'{name}.png')
    plt.close(figure)


view = (energy >= max(0., origins.min()-.08)) & (energy <= vertical_energy.max()*HARTREE_TO_EV+1.)
profile = f'B3LYP/def2-SVP · grid level 3 · 0 K VG · Gaussian FWHM {SETTINGS["fwhm_cm1"]:g} cm⁻¹'
fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), constrained_layout=True)
axes[0].plot(energy[view], per_ev[view, 0], label='Franck–Condon sum', lw=1.7, color='#2979b8')
axes[0].plot(energy[view], per_ev[view, 3], label='FC + linear HT sum', lw=1.7, color='#d44d38')
axes[0].set(ylabel='Oscillator-strength density / eV⁻¹', title=f'Anthracene: lowest {nstates} singlet absorption bands')
axes[0].legend(frameon=False)
axes[1].plot(energy[view], per_ev[view, 1], label='HT sum', color='#267d52')
axes[1].plot(energy[view], per_ev[view, 2], label='Signed within-state interference sum', color='#8655a1')
axes[1].axhline(0, color='0.4', lw=.6)
axes[1].set(ylabel='Contribution / eV⁻¹')
axes[1].legend(frameon=False)
for state in range(nstates):
    axes[2].plot(energy[view], per_state[state, view, 3], lw=1.3, label=f'S{state+1}')
axes[2].set(xlabel='Photon energy / eV', ylabel='Absolute state density / eV⁻¹')
axes[2].legend(frameon=False, ncol=min(nstates, 4))
for ax in axes[:2]:
    ax.set_xlabel('Photon energy / eV')
fig.suptitle(profile, fontsize=12)
save_figure(fig, 'anthracene_fc_ht')

fig, panels = plt.subplots((nstates+3)//4, min(4, nstates), figsize=(15, 3.4*((nstates+3)//4)),
                           squeeze=False, constrained_layout=True)
for state, ax in enumerate(panels.ravel()):
    if state >= nstates:
        ax.set_visible(False)
        continue
    band = (energy >= origins[state]-.05) & (energy <= vertical_energy[state]*HARTREE_TO_EV+.8)
    ax.plot(energy[band], per_state[state, band, 0], color='#2979b8', lw=1.4, label='FC')
    ax.plot(energy[band], per_state[state, band, 3], color='#d44d38', lw=1.4, label='FC + HT')
    ax.axvline(origins[state], color='0.4', ls=':', lw=1., label='Model 0–0')
    ax.set(title=f'S₀ → S{state+1}', xlabel='Photon energy / eV', ylabel='Density / eV⁻¹')
    ax.legend(frameon=False, fontsize=8)
fig.suptitle('Absolute state spectra; panel ranges differ, intensities are not normalized', fontsize=12)
save_figure(fig, 'anthracene_fc_ht_states')

fig, ax = plt.subplots(figsize=(10, max(3., nstates*.45)), constrained_layout=True)
image = ax.imshow(hr, origin='lower', aspect='auto', cmap='YlGnBu',
                  extent=(.5, nmodes+.5, .5, nstates+.5))
ax.set(xlabel='Normal mode index', ylabel='Electronic state', yticks=roots,
       yticklabels=[f'S{root}' for root in roots], title='VG displacement: Huang–Rhys factors for every retained mode')
fig.colorbar(image, ax=ax, label='Huang–Rhys factor')
save_figure(fig, 'anthracene_mode_huang_rhys')

fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
ax.vlines(vertical_energy*HARTREE_TO_EV, 0, vertical_strengths, color='#2979b8', lw=2)
for root, value, strength in zip(roots, vertical_energy, vertical_strengths):
    ax.annotate(f'S{root}', (value*HARTREE_TO_EV, strength), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=9)
ax.margins(y=.2)
ax.set(xlabel='Vertical excitation energy / eV', ylabel='Oscillator strength',
       title=f'Anthracene: {nstates} target full-TDDFT singlets; guard roots excluded')
save_figure(fig, 'anthracene_vertical')

save('spectrum_status.json', dict(complete=True, state_count=nstates, mode_count=nmodes,
    target_roots=roots.tolist(), vertical_roots_plotted=roots.tolist(), signature=signature,
    mode_response_sha256=hashlib.sha256((OUTPUT/'mode_response.npz').read_bytes()).hexdigest(),
    model='0 K equal-curvature VG + linear HT; incoherent sum of independent singlet bands',
    checks=dict(per_state=checks, interpolation_relative_area_error=area_errors.tolist(),
        summed_relative_area_error=summed_error, area_tolerance=AREA_TOLERANCE,
        common_energy_grid_step_ev=step*HARTREE_TO_EV),
    integrated_oscillator_strengths=summed_area.tolist(),
    per_state_integrated_oscillator_strengths=state_areas.tolist(),
    thermal_populations=False, duschinsky=False, electronic_state_coupling=False,
    empirical_energy_shift_ev=0., derivative_method='Complete displaced SCF and full TDDFT central differences',
    figures=figures))
