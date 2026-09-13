"""Plot archived Gamma-point velocity-gauge spectra without recalculation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    summary=json.loads((root/'summary.json').read_text());data=np.load(root/'spectrum.npz')
    grid=data['grid_ev'];ours=data['gradscf_spectrum'];reference=data['pyscf_spectrum']
    difference=ours-reference
    fig,axes=plt.subplots(3,1,figsize=(11,8.3),sharex=True,
        gridspec_kw={'height_ratios':[3.2,1.1,1.1]})
    ax,sticks,error=axes
    ax.fill_between(grid,0,ours,color='#1764a3',alpha=.08)
    ax.plot(grid,reference,color='#cf792e',lw=2.2,ls=(0,(5,3)),label='PySCF')
    ax.plot(grid,ours,color='#1764a3',lw=1.25,label='GradSCF')
    ax.set_ylabel('Oscillator-strength density\n(eV⁻¹)')
    ax.set_ylim(0,max(ours.max(),reference.max())*1.13)
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles[::-1],labels[::-1],loc='upper right',frameon=False)
    groups=summary['groups']
    bright=[g for g in groups if g['pyscf_strength']>1e-6]
    if bright and bright[-1]['pyscf_energy_ev']>10:
        center=bright[-1]['pyscf_energy_ev']
        inset=ax.inset_axes([.52,.20,.43,.53])
        selected=(grid>center-1)&(grid<center+1)
        inset.plot(grid[selected],reference[selected],color='#cf792e',lw=2,ls='--')
        inset.plot(grid[selected],ours[selected],color='#1764a3',lw=1.1)
        inset.set_xlim(center-1,center+1);inset.set_ylim(0,max(reference[selected])*1.15)
        inset.set_title('High-energy zoom (same units)',fontsize=9)
        inset.tick_params(labelsize=8,direction='out');inset.grid(color='#e5e9ed',lw=.5)
        for side in ['top','right']:inset.spines[side].set_visible(True)
        for spine in inset.spines.values():spine.set_color('#a9b4bf')
    for group in groups:
        if max(group['gradscf_strength'],group['pyscf_strength'])<1e-8:continue
        eg,er=group['gradscf_energy_ev'],group['pyscf_energy_ev']
        fg,fr=group['gradscf_strength'],group['pyscf_strength']
        sticks.vlines(er,0,fr,color='#cf792e',lw=3,alpha=.85)
        sticks.plot(er,fr,'o',mfc='white',mec='#cf792e',ms=5)
        sticks.vlines(eg,0,fg,color='#1764a3',lw=1)
        sticks.plot(eg,fg,'.',color='#1764a3',ms=4)
    sticks.set_ylim(bottom=0);sticks.set_ylabel('Grouped oscillator\nstrength f')
    error.plot(grid,difference,color='#385a73',lw=1)
    error.axhline(0,color='#9aa5b0',lw=.7,ls='--')
    error.set_ylabel('GradSCF − PySCF\n(eV⁻¹)')
    error.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    lim=max(np.max(np.abs(difference))*1.15,1e-14);error.set_ylim(-lim,lim)
    error.set_xlabel('Photon energy (eV)')
    for panel in axes:
        panel.set_xlim(grid[0],grid[-1]);panel.tick_params(direction='out',length=3)
        panel.grid(axis='x',color='#dce2e7',lw=.6)
    fig.suptitle('Periodic Si: velocity-gauge TDDFT spectrum',x=.09,ha='left',y=.978,fontsize=18,weight='bold')
    fig.text(.09,.927,f"Γ point  •  PBE / GTH-SZV / GTH-PBE  •  FFT {summary['fft_mesh'][0]}³  •  {summary['nstates']} singlet states",fontsize=11,color='#46515c')
    fig.text(.09,.89,f"Gaussian FWHM = {summary['fwhm_ev']:.2f} eV  |  Peak-normalized max difference = {summary['relative_peak_spectrum_error']:.2e}",fontsize=10,color='#46515c')
    fig.text(.09,.025,'Unit-area broadening; absolute oscillator strengths retained. Both include the nonlocal GTH velocity correction.\n'
        'Γ-point per-cell spectrum with degenerate-state grouping. Bulk k integration and macroscopic absorption coefficients are not computed.',
        fontsize=8.5,color='#46515c',va='bottom')
    fig.subplots_adjust(left=.105,right=.97,top=.85,bottom=.13,hspace=.22)
    for extension in ['png','pdf','svg']:
        fig.savefig(root/f'absorption_comparison.{extension}',dpi=230,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
