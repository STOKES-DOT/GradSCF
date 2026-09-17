"""Render archived band comparisons without rerunning electronic structure."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EV=27.211386245988
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':14,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory
    summaries=json.loads((root/'summary.json').read_text())
    fig,axes=plt.subplots(2,len(summaries),figsize=(13,7.8),squeeze=False,
        gridspec_kw={'height_ratios':[3.5,1.25]})
    names={'silicon':'Silicon (Si)','diamond':'Diamond (C)'}
    largest=max(s['max_band_error_ev'] for s in summaries)
    error_scale,error_unit=(1e9,'neV') if largest<1e-8 else ((1e6,'μeV') if largest<1e-4 else (1e3,'meV'))
    for col,s in enumerate(summaries):
        data=np.load(root/s['system']/'bands.npz')
        x=data['distance'];ticks=x[data['tick_indices']];zero=float(data['reference_zero_hartree'])
        ours=(data['gradscf_hartree']-zero)*EV;ref=(data['pyscf_hartree']-zero)*EV
        error=(data['gradscf_hartree']-data['pyscf_hartree'])*EV*error_scale
        ax,err=axes[:,col]
        low=np.floor(min(ours.min(),ref.min())/5)*5-1
        high=np.ceil(max(ours.max(),ref.max())/5)*5+1
        ax.axhspan(low,0,color='#f0f5fa',zorder=0)
        for band in range(ours.shape[1]):
            ax.plot(x,ref[:,band],color='#cf792e',lw=2,ls=(0,(5,3)),alpha=.85)
            ax.plot(x,ours[:,band],color='#1764a3',lw=1.05)
            err.plot(x,error[:,band],lw=.95,color=plt.cm.viridis((band+.5)/ours.shape[1]),alpha=.9)
        for panel in [ax,err]:
            for tick in ticks:panel.axvline(tick,color='#87929e',lw=.6,alpha=.45,zorder=0)
            panel.axhline(0,color='#56616d',lw=.7,ls='--')
            panel.set_xlim(x[0],x[-1]);panel.set_xticks(ticks)
            panel.set_xticklabels([r'$\Gamma$' if label=='G' else label for label in s['kpath']])
            panel.tick_params(direction='out',length=3)
        ax.set_ylim(low,high);ax.set_ylabel(r'$E-E_{\mathrm{VBM}}^{\mathrm{PySCF}}$ (eV)')
        ax.set_title(names[s['system']]+f"  |  a = {s['lattice_constant_angstrom']:.3f} Å",loc='left',pad=30)
        note=(f"Sampled gap: {s['gradscf_sampled_gap_ev']:.4f} / {s['pyscf_sampled_gap_ev']:.4f} eV\n"
              f"Max |ΔE| = {s['max_band_error_ev']:.2e} eV")
        ax.text(0.,1.015,note.replace('\n','  |  '),transform=ax.transAxes,ha='left',va='bottom',fontsize=9,color='#46515c')
        lim=max(np.max(np.abs(error))*1.18,1e-6)
        err.set_ylim(-lim,lim);err.set_ylabel(f'GradSCF − PySCF\n({error_unit})')
        err.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
        err.set_xlabel('Wave-vector path (distance in reciprocal space)')
    first=summaries[0]
    fig.suptitle('Periodic PBE bands: GradSCF vs PySCF',x=.065,ha='left',fontsize=19,weight='bold',y=.98)
    subtitle=(f"GTH-SZV / GTH-PBE  •  SCF mesh {first['scf_kmesh'][0]}×{first['scf_kmesh'][1]}×{first['scf_kmesh'][2]}  •  "
              f"FFT {first['fft_mesh'][0]}³  •  {first['n_path_points']} path points, {first['n_bands']} bands")
    fig.text(.065,.929,subtitle,color='#46515c',fontsize=11)
    fig.legend(handles=[Line2D([0],[0],color='#1764a3',lw=1.4,label='GradSCF'),
                        Line2D([0],[0],color='#cf792e',lw=2,ls='--',label='PySCF')],
               loc='upper right',bbox_to_anchor=(.96,.94),ncol=2,frameon=False)
    fig.text(.065,.019,'One shared PySCF VBM zero per material. Sampled gap values: GradSCF / PySCF.\n'
             'Minimal basis and coarse SCF k mesh; basis and k-mesh convergence of the sampled gaps remains untested.',
             fontsize=9,color='#46515c',va='bottom')
    fig.subplots_adjust(left=.075,right=.97,top=.82,bottom=.14,hspace=.22,wspace=.24)
    for extension in ['png','pdf','svg']:
        fig.savefig(root/f'band_comparison.{extension}',dpi=220,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
