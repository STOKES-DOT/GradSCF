"""Plot the existing torsion data on absolute, unshifted SCF energy axes."""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np

OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

source=OUT/'benchmark-curves.csv'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
rows=list(csv.DictReader(source.open()))
names=['STO-3G','3-21G','6-31G(d,p)','cc-pVDZ','cc-pVTZ',
       'NNAO / MACE','NNAO / free contraction']
colors=['#A4A8AC','#68717A','#7C8938','#B95F83','#222B34','#D87632','#28699C']
styles=[(':','x'),('--','v'),('-.','^'),('--','D'),('-',None),('-','o'),('--','s')]
assert len(rows)==175 and {r['basis'] for r in rows}==set(names)
curves={}
summary=[]
for name,color,style in zip(names,colors,styles):
    selected=sorted([r for r in rows if r['basis']==name],key=lambda r:float(r['angle_deg']))
    angle=np.array([float(r['angle_deg']) for r in selected])
    energy=np.array([float(r['energy_hartree']) for r in selected])
    np.testing.assert_array_equal(angle,np.arange(0,121,5))
    assert np.isfinite(energy).all()
    curves[name]=dict(angle=angle,energy=energy,color=color,style=style)
    summary.append(dict(basis=name,nao=int(selected[0]['nao']),energy_0_hartree=float(energy[0]),
        energy_60_hartree=float(energy[12]),minimum_energy_hartree=float(energy.min()),
        maximum_energy_hartree=float(energy.max())))
with (OUT/'absolute-energy-summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#687078',
    'text.color':'#24292F','axes.labelcolor':'#24292F'})
fig=plt.figure(figsize=(12.8,8.3))
grid=fig.add_gridspec(2,2,width_ratios=[1,1.12],hspace=.36,wspace=.40)
overview=fig.add_subplot(grid[:,0])
polarized=fig.add_subplot(grid[0,1])
nnao=fig.add_subplot(grid[1,1])
panels=[(overview,names,'All seven curves'),
        (polarized,names[2:],'Zoom: polarized bases and NNAO'),
        (nnao,names[-2:],'Zoom: NNAO absolute energies')]
for ax,selected,title in panels:
    for name in selected:
        curve=curves[name]
        linestyle,marker=curve['style']
        ax.plot(curve['angle'],curve['energy'],color=curve['color'],linestyle=linestyle,
            marker=marker,markersize=4.1,markevery=2,
            linewidth=2.1 if name.startswith('NNAO') or name=='cc-pVTZ' else 1.35,label=name)
    values=np.concatenate([curves[name]['energy'] for name in selected])
    padding=.06*np.ptp(values)
    ax.set_ylim(values.min()-padding,values.max()+padding)
    ax.set_title(title,pad=10,fontsize=11.5)
    ax.set_ylabel('SCF total energy (Eh)')
    ax.set_xlabel('Methyl rotation (degrees)')
    ax.set_xlim(-3,123);ax.set_xticks(np.arange(0,121,30))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f' if ax is nnao else '%.3f'))
    ax.set_axisbelow(True);ax.grid(color='#E5E8EA',lw=.7)
    ax.axvline(60,color='#A9B0B5',lw=.8,ls=':')
handles,labels=overview.get_legend_handles_labels()
fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.53,.92),
    frameon=False,fontsize=9.7,columnspacing=2.)
fig.suptitle('Ethane torsion: absolute SCF energies',fontsize=15,y=.98)
fig.text(.075,.025,'25 rigid geometries, 5-degree spacing; DF-RHF / def2-universal-jkfit; MACE update 3351.\n'
    'Every panel shows the original total energies in Hartree. No per-curve energy shift is applied; right panels only change the axis range.',
    fontsize=8.5,color='#59636B')
fig.subplots_adjust(left=.075,right=.98,bottom=.13,top=.77)
fig.savefig(OUT/'ethane-torsion-absolute.png',dpi=200,facecolor='white')
fig.savefig(OUT/'ethane-torsion-absolute.pdf',facecolor='white')
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
(OUT/'absolute-plot-provenance.json').write_text(json.dumps(dict(
    input_file=source.name,input_sha256=source_hash,energy_unit='hartree',energy_shift_applied=False,
    curve_count=7,points_per_curve=25,recomputed_scf=False,summary=summary),indent=2)+'\n')
print(json.dumps(summary,indent=2))
