"""Compare the partial p-valence expansion with the previous seven curves."""
from pathlib import Path
import csv
import json
import os
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'nnao-ethane-torsion-20261018'
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

scan=json.loads((OUT/'summary.json').read_text())
checked=json.loads((OUT/'gpu4-check.json').read_text())
assert scan['status']=='complete' and len(scan['points'])==25
assert checked['status']=='complete' and checked['passed']==25
points=sorted(scan['points'],key=lambda r:r['angle_deg'])
angle=np.array([r['angle_deg'] for r in points])
np.testing.assert_array_equal(angle,np.arange(0,121,5))
assert all(r['best_gradient_converged'] for r in points)
mapped=[json.loads((OUT/f"angle-{int(r['angle_deg']):03d}"/'summary.json').read_text())
        for r in points if r['optimization_mode']=='exact_symmetry_transfer_with_scf_and_gradient_check']
old_rows=list(csv.DictReader((BASE/'benchmark-curves.csv').open()))
names=['STO-3G','3-21G','6-31G(d,p)','cc-pVDZ','cc-pVTZ','NNAO / MACE',
       'NNAO / free contraction','NNAO / C p2 / free']
colors=['#A4A8AC','#68717A','#7C8938','#B95F83','#222B34','#D87632','#28699C','#7256A8']
styles=[(':','x'),('--','v'),('-.','^'),('--','D'),('-',None),('-','o'),('--','s'),('-.','P')]
curves={}
for name,color,style in zip(names,colors,styles):
    if name==names[-1]:
        energy=np.array([r['best_energy_hartree'] for r in points]);nao=50
    else:
        rows=sorted([r for r in old_rows if r['basis']==name],key=lambda r:float(r['angle_deg']))
        np.testing.assert_array_equal([float(r['angle_deg']) for r in rows],angle)
        energy=np.array([float(r['energy_hartree']) for r in rows]);nao=int(rows[0]['nao'])
    curves[name]=dict(energy=energy,relative=(energy-energy[0])*627.5094740631,
                      nao=nao,color=color,style=style)
old=curves[names[-2]];new=curves[names[-1]];reference=curves['cc-pVTZ']
gain=(old['energy']-new['energy'])*627.5094740631
assert np.all(gain>=-1e-6)
metrics=dict(old_nao=44,new_nao=50,pool_nao=152,points=25,fixed_exponents=True,
    energy_lowering_0_kcal_mol=float(gain[0]),energy_lowering_60_kcal_mol=float(gain[12]),
    energy_lowering_min_kcal_mol=float(gain.min()),energy_lowering_max_kcal_mol=float(gain.max()),
    old_barrier_kcal_mol=float(np.ptp(old['relative'])),new_barrier_kcal_mol=float(np.ptp(new['relative'])),
    barrier_change_kcal_mol=float(np.ptp(new['relative'])-np.ptp(old['relative'])),
    ccpvtz_barrier_kcal_mol=float(np.ptp(reference['relative'])),
    old_max_curve_difference_vs_ccpvtz_kcal_mol=float(np.abs(old['relative']-reference['relative']).max()),
    new_max_curve_difference_vs_ccpvtz_kcal_mol=float(np.abs(new['relative']-reference['relative']).max()),
    old_rms_curve_difference_vs_ccpvtz_kcal_mol=float(np.sqrt(np.mean((old['relative']-reference['relative'])**2))),
    new_rms_curve_difference_vs_ccpvtz_kcal_mol=float(np.sqrt(np.mean((new['relative']-reference['relative'])**2))),
    closure_error_hartree=float(abs(new['energy'][-1]-new['energy'][0])),
    mirror_max_error_hartree=float(np.abs(new['energy']-new['energy'][::-1]).max()),
    gradient_converged_points=sum(r['best_gradient_converged'] for r in points),
    gradient_converged_starts=sum(r['converged_starts'] for r in points),
    independently_optimized_points=sum(r['optimization_mode']=='independent_two_start_optimization' for r in points),
    symmetry_transferred_checked_points=sum(r['optimization_mode']=='exact_symmetry_transfer_with_scf_and_gradient_check' for r in points),
    maximum_mapping_geometry_error_angstrom=max(r['mapping_geometry_error_angstrom'] for r in mapped),
    maximum_mapping_energy_error_hartree=max(r['mapped_energy_error_hartree'] for r in mapped),
    maximum_mapping_gradient_error=max(r['mapped_gradient_max_error'] for r in mapped),
    maximum_start_spread_hartree=max(r['endpoint_spread_hartree'] for r in points if r['endpoint_spread_hartree'] is not None),
    maximum_gradient_norm=max(r['best_gradient_norm'] for r in points),
    maximum_independent_energy_error_hartree=max(r['native_energy_error_hartree'] for r in checked['results']),
    elapsed_seconds=scan['elapsed_seconds'],job_id=scan['job_id'])
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
records=[];summary=[]
for name,c in curves.items():
    summary.append(dict(basis=name,nao=c['nao'],energy_0_hartree=float(c['energy'][0]),
        energy_60_hartree=float(c['energy'][12]),grid_barrier_kcal_mol=float(np.ptp(c['relative'])),
        max_curve_difference_vs_ccpvtz_kcal_mol=float(np.abs(c['relative']-reference['relative']).max())))
    for a,e,r in zip(angle,c['energy'],c['relative']):
        records.append(dict(basis=name,angle_deg=float(a),energy_hartree=float(e),
                            relative_energy_kcal_mol=float(r),nao=c['nao']))
for filename,rows in [('curves.csv',records),('basis-summary.csv',summary)]:
    with (OUT/filename).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#687078',
    'text.color':'#24292F','axes.labelcolor':'#24292F'})
fig,axes=plt.subplots(2,1,figsize=(10.2,8.6),sharex=True,
    gridspec_kw={'height_ratios':[1.8,1],'hspace':.13})
for name,c in curves.items():
    linestyle,marker=c['style']
    width=2.2 if name.startswith('NNAO') or name=='cc-pVTZ' else 1.3
    for ax,y in [(axes[0],c['relative']),(axes[1],c['relative']-reference['relative'])]:
        ax.plot(angle,y,color=c['color'],linestyle=linestyle,marker=marker,markevery=2,
            markersize=4.5,linewidth=width,label=f"{name} ({c['nao']} AOs)")
axes[0].set_ylabel(r'$E(\phi)-E(0^\circ)$ (kcal/mol)')
axes[0].set_ylim(bottom=-.05)
axes[0].legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.35),fontsize=9.)
axes[1].set_ylabel('Relative-curve difference\nfrom HF/cc-pVTZ (kcal/mol)')
axes[1].set_xlabel('Methyl rotation from the initial staggered geometry (degrees)')
axes[1].set_xticks(np.arange(0,121,15));axes[1].set_xlim(-2,122)
axes[1].axhline(0,color='#424A52',lw=.8)
for ax in axes:
    ax.set_axisbelow(True);ax.grid(color='#E5E8EA',lw=.7)
    ax.axvline(60,color='#A9B0B5',lw=.8,ls=':')
fig.suptitle('Ethane torsion: adding a second carbon valence-p radial channel',fontsize=14,y=.98)
fig.text(.13,.025,'25 rigid geometries; DF-RHF / def2-universal-jkfit. Each relative curve uses its own 0-degree energy.\n'
    'New curve: C [2s2p1d], H [1s1p], free contractions; unchanged exponents, core and primitive pool. MACE remains the 44-AO checkpoint.',
    fontsize=8.,color='#59636B')
fig.subplots_adjust(left=.13,right=.98,bottom=.12,top=.79)
fig.savefig(OUT/'ethane-p2-relative.png',dpi=200,facecolor='white')
fig.savefig(OUT/'ethane-p2-relative.pdf',facecolor='white')
plt.close(fig)

fig=plt.figure(figsize=(12.8,8.3))
grid=fig.add_gridspec(2,2,width_ratios=[1,1.12],hspace=.36,wspace=.40)
overview=fig.add_subplot(grid[:,0]);polarized=fig.add_subplot(grid[0,1]);nnao=fig.add_subplot(grid[1,1])
for ax,selected,title in [(overview,names,'All eight curves'),
        (polarized,names[2:],'Zoom: polarized bases and NNAO'),
        (nnao,names[-3:],'Zoom: NNAO absolute energies')]:
    for name in selected:
        c=curves[name];linestyle,marker=c['style']
        ax.plot(angle,c['energy'],color=c['color'],linestyle=linestyle,marker=marker,markevery=2,
            markersize=4.1,linewidth=2.1 if name.startswith('NNAO') or name=='cc-pVTZ' else 1.35,label=name)
    values=np.concatenate([curves[name]['energy'] for name in selected]);padding=.06*np.ptp(values)
    ax.set_ylim(values.min()-padding,values.max()+padding)
    ax.set_title(title,pad=10,fontsize=11.5);ax.set_ylabel('SCF total energy (Eh)')
    ax.set_xlabel('Methyl rotation (degrees)');ax.set_xlim(-3,123);ax.set_xticks(np.arange(0,121,30))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f' if ax is nnao else '%.3f'))
    ax.set_axisbelow(True);ax.grid(color='#E5E8EA',lw=.7);ax.axvline(60,color='#A9B0B5',lw=.8,ls=':')
handles,labels=overview.get_legend_handles_labels()
fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.53,.92),frameon=False,fontsize=9.4)
fig.suptitle('Ethane torsion: absolute energies with a second carbon valence-p channel',fontsize=14,y=.98)
fig.text(.075,.025,'25 rigid geometries, 5-degree spacing; DF-RHF / def2-universal-jkfit. All panels show unshifted total energies in Hartree.\n'
    'New 50-AO curve: same pool and core; free contractions with checked symmetry reuse. MACE retains the original 44-AO checkpoint.',
    fontsize=8.3,color='#59636B')
fig.subplots_adjust(left=.075,right=.98,bottom=.13,top=.77)
fig.savefig(OUT/'ethane-p2-absolute.png',dpi=200,facecolor='white')
fig.savefig(OUT/'ethane-p2-absolute.pdf',facecolor='white')

lines=['# Ethane: second carbon valence-p radial channel','',
    'Only the C valence-p radial multiplicity changes: [2s1p1d] -> [2s2p1d]. '
    'H remains [1s1p]. The compact AO count increases from 44 to 50; the mixed primitive pool remains 152. '
    'All exponents, fixed core basis functions, geometry, DF approximation and SCF tolerances are held fixed.',
    'The added curve is free contraction only; the existing 44-AO MACE checkpoint is unchanged.',
    '', '| Quantity | Value |','| --- | ---: |']
for key in ['energy_lowering_0_kcal_mol','energy_lowering_60_kcal_mol',
            'old_barrier_kcal_mol','new_barrier_kcal_mol','barrier_change_kcal_mol','ccpvtz_barrier_kcal_mol',
            'old_max_curve_difference_vs_ccpvtz_kcal_mol','new_max_curve_difference_vs_ccpvtz_kcal_mol',
            'gradient_converged_points','independently_optimized_points','symmetry_transferred_checked_points',
            'gradient_converged_starts','maximum_start_spread_hartree',
            'maximum_independent_energy_error_hartree','elapsed_seconds']:
    lines.append(f'| {key} | {metrics[key]:.10g} |')
lines+=['','RHF energies use native cached S/h/packed DF and existing implicit SCF gradients. '
    'The first half-period and any already completed second-half points use independent two-start optimization. '
    'Unfinished equivalent angles reuse coefficients through an exact reflected atom map, '
    'followed by a new SCF energy and full coefficient gradient evaluation at the target geometry. '
    'No curve interpolation is used. All 25 final frozen bases were independently checked with GPU4PySCF. '
    'Finite-budget optimizations from two starts do not prove global minima. '
    'HF/cc-pVTZ is a comparison basis, not an exact reference. All barriers are from the rigid scan.',
    '',f"Slurm jobs {scan.get('previous_job_id','')}, {scan['job_id']}, {scan['device']}, 32 CPU; "
    f"total runtime {scan['elapsed_seconds']:.2f} seconds. "
    'No finite-difference validation was performed. See RUN_SPEC.md and configuration.json for settings.',
    '', 'Plots: `ethane-p2-relative.png/.pdf`, `ethane-p2-absolute.png/.pdf`. '
    'Raw curves: `curves.csv`; per-angle checkpoints and histories: `angle-*/`.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(metrics,indent=2))
