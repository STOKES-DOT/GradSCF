"""The S1 VG/linear-HT absorption envelope: FC, HT, interference and total.

All 66 modes are retained in the analytic oscillator product. The Gaussian
width is a display convolution, not a lifetime or finite-temperature model.
"""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from common import OUTPUT, ROOT, SETTINGS, save
from gradscf_tools.vibronic import spectrum

HARTREE_TO_EV = 27.211386245988
CM1 = 219474.6313631965
FIGURES = OUTPUT.parent

if not json.loads((OUTPUT/'mode_response_status.json').read_text()).get('complete'):
    raise RuntimeError('Validated complete normal-mode derivatives are required.')
data = np.load(OUTPUT/'mode_response.npz')
axis, density, checks = spectrum(float(data['vertical_energy']),data['frequencies'],
    data['gap_gradient'],data['dipole'],data['dipole_derivative'],
    fwhm_cm1=SETTINGS['fwhm_cm1'],ntime=32768,time_step=10.)
energy, per_ev = axis*HARTREE_TO_EV,density/HARTREE_TO_EV
np.savez(OUTPUT/'spectrum.npz',energy_ev=energy,oscillator_density_per_ev=per_ev,
         columns=['fc','ht','interference','total'])
np.savetxt(FIGURES/'spectrum.csv',np.column_stack((energy,per_ev)),delimiter=',',
    header='energy_eV,FC_per_eV,HT_per_eV,interference_per_eV,total_per_eV',comments='')
hr = data['gap_gradient']**2/(2*data['frequencies']**3)
with (FIGURES/'modes.csv').open('w') as stream:
    writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['mode','frequency_cm-1','Huang_Rhys','d_gap_dQ_me','d_mu_x_dQ_me','d_mu_y_dQ_me','d_mu_z_dQ_me'])
    for mode in range(len(hr)):
        writer.writerow([mode+1,data['frequencies'][mode]*CM1,hr[mode],data['gap_gradient'][mode],*data['dipole_derivative'][:,mode]])
origin=checks['zero_zero_energy_hartree']*HARTREE_TO_EV
window=(energy>=origin-.04)&(energy<=origin+1.)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,1,figsize=(9,10),constrained_layout=True)
axes[0].plot(energy[window],per_ev[window,0],label='Franck–Condon',lw=1.6,color='#2979b8')
axes[0].plot(energy[window],per_ev[window,3],label='FC + linear HT',lw=1.6,color='#d44d38')
axes[0].axvline(origin,color='0.4',ls=':',lw=1,label='Model 0–0')
axes[0].set(xlabel='Photon energy / eV',ylabel='Oscillator-strength density / eV⁻¹',
            title='Anthracene S₀ → S₁: FC versus FC + HT')
axes[0].legend(frameon=False)
axes[1].plot(energy[window],per_ev[window,1],label='HT',color='#267d52')
axes[1].plot(energy[window],per_ev[window,2],label='Signed interference',color='#8655a1')
axes[1].axhline(0,color='0.4',lw=.6)
axes[1].set(xlabel='Photon energy / eV',ylabel='Contribution / eV⁻¹')
axes[1].legend(frameon=False)
axes[2].bar(data['frequencies']*CM1,hr,width=13,color='#2979b8')
axes[2].set(xlabel='Mode wavenumber / cm⁻¹',ylabel='Huang–Rhys factor',title='All 66 normal modes retained')
fig.suptitle(f'B3LYP/def2-SVP · grid level 3 · 0 K VG model · Gaussian FWHM {SETTINGS["fwhm_cm1"]:g} cm⁻¹',fontsize=12)
for suffix in ('png','pdf'):
    fig.savefig(FIGURES/f'anthracene_fc_ht.{suffix}',dpi=220)
plt.close(fig)
vertical=np.load(OUTPUT/'vertical.npz')
fig,ax=plt.subplots(figsize=(8,3.8),constrained_layout=True)
ax.vlines(vertical['excitation']*HARTREE_TO_EV,0,vertical['oscillator_strength'],color='#2979b8',lw=2)
ax.set(xlabel='Vertical excitation energy / eV',ylabel='Oscillator strength',title='Anthracene: eight full-TDDFT singlets at the S₀ minimum')
for suffix in ('png','pdf'):
    fig.savefig(FIGURES/f'anthracene_vertical.{suffix}',dpi=220)
plt.close(fig)
save('spectrum_status.json',dict(complete=True,model='0 K equal-curvature VG + linear HT; S1 band only',
    checks=checks,integrated_oscillator_strengths=trapezoid(per_ev,energy,axis=0).tolist(),
    thermal_populations=False,duschinsky=False,empirical_energy_shift_ev=0.,
    derivative_method='Complete displaced SCF and full TDDFT central differences',
    figures=['anthracene_fc_ht.png','anthracene_vertical.png']))
