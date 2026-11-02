"""Overlay all seven relative torsional curves and compare grid barriers."""
from pathlib import Path
import csv
import json
import os
import numpy as np

OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scan=json.loads((OUT/'scan-summary.json').read_text())
traditional=json.loads((OUT/'traditional-results.json').read_text())
checks=json.loads((OUT/'gpu4-check.json').read_text())
assert scan['status']=='complete' and len(scan['points'])==25
assert traditional['status']=='complete' and traditional['passed']==125
assert checks['status']=='complete' and checks['passed']==50
factor=627.5094740631
curves={}
for name in ['sto-3g','3-21g','6-31g(d,p)','cc-pvdz','cc-pvtz']:
    rows=sorted([r for r in traditional['results'] if r['basis']==name],key=lambda r:r['angle_deg'])
    assert len(rows)==25 and all(r['status']=='passed' for r in rows)
    angle=np.array([r['angle_deg'] for r in rows])
    np.testing.assert_array_equal(angle,np.arange(0,121,5))
    energy=np.array([r['energy_hartree'] for r in rows])
    curves[name]=dict(energy=energy,relative=(energy-energy[0])*factor,nao=rows[0]['nao'],
        backend='GPU4PySCF',color={'sto-3g':'#A4A8AC','3-21g':'#68717A',
        '6-31g(d,p)':'#7C8938','cc-pvdz':'#B95F83','cc-pvtz':'#222B34'}[name])
for name,key,color in [('NNAO MACE','mace_energy_hartree','#D87632'),
                        ('NNAO free','free_energy_hartree','#28699C')]:
    rows=sorted(scan['points'],key=lambda r:r['angle_deg'])
    np.testing.assert_array_equal([r['angle_deg'] for r in rows],angle)
    energy=np.array([r[key] for r in rows])
    curves[name]=dict(energy=energy,relative=(energy-energy[0])*factor,nao=rows[0]['nao'],
                      backend='GradSCF; same-basis GPU4PySCF checked',color=color)

labels={'sto-3g':'STO-3G','3-21g':'3-21G','6-31g(d,p)':'6-31G(d,p)',
        'cc-pvdz':'cc-pVDZ','cc-pvtz':'cc-pVTZ','NNAO MACE':'NNAO / MACE','NNAO free':'NNAO / free contraction'}
lookup60=int(np.flatnonzero(angle==60)[0])
reference=curves['cc-pvtz']['relative']
summary=[];long=[]
for name,c in curves.items():
    relative=c['relative'];energy=c['energy']
    summary.append(dict(basis=labels[name],nao=c['nao'],zero_energy_hartree=float(energy[0]),
        grid_barrier_kcal_mol=float(np.ptp(relative)),e60_minus_e0_kcal_mol=float(relative[lookup60]),
        grid_maximum_angle_deg=float(angle[np.argmax(relative)]),
        max_curve_difference_vs_ccpvtz_kcal_mol=float(np.abs(relative-reference).max()),
        rms_curve_difference_vs_ccpvtz_kcal_mol=float(np.sqrt(np.mean((relative-reference)**2))),
        closure_error_hartree=float(abs(energy[-1]-energy[0])),
        mirror_max_error_hartree=float(np.abs(energy-energy[::-1]).max()),backend=c['backend']))
    for a,e,r in zip(angle,energy,relative):
        long.append(dict(basis=labels[name],angle_deg=float(a),energy_hartree=float(e),
                         relative_energy_kcal_mol=float(r),nao=c['nao']))
for filename,rows in [('benchmark-summary.csv',summary),('benchmark-curves.csv',long)]:
    with (OUT/filename).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'benchmark-summary.json').write_text(json.dumps(dict(method='DF-RHF',auxbasis='def2-universal-jkfit',
    checkpoint_update=3351,angles_degrees=angle.tolist(),traditional_job_id=traditional['job_id'],
    traditional_seconds=traditional['elapsed_seconds'],results=summary),indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,
    'pdf.fonttype':42,
    'axes.spines.right':False,'axes.edgecolor':'#687078','text.color':'#24292F','axes.labelcolor':'#24292F'})
fig,axes=plt.subplots(2,1,figsize=(9.6,8.5),sharex=True,
    gridspec_kw={'height_ratios':[1.8,1],'hspace':.13})
styles={'sto-3g':(':','x'),'3-21g':('--','v'),'6-31g(d,p)':('-.','^'),
        'cc-pvdz':('--','D'),'cc-pvtz':('-',None),'NNAO MACE':('-','o'),'NNAO free':('--','s')}
for name,c in curves.items():
    linestyle,marker=styles[name]
    width=2.2 if name.startswith('NNAO') or name=='cc-pvtz' else 1.3
    for ax,y in [(axes[0],c['relative']),(axes[1],c['relative']-reference)]:
        ax.plot(angle,y,color=c['color'],linestyle=linestyle,marker=marker,markevery=2,
            markersize=4.5,linewidth=width,label=f"{labels[name]} ({c['nao']} AOs)")
axes[0].set_ylabel(r'$E(\phi)-E(0^\circ)$ (kcal/mol)')
axes[0].set_ylim(bottom=-.05)
axes[0].legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.35),fontsize=9.1)
axes[1].set_ylabel('Relative-curve difference\nfrom HF/cc-pVTZ (kcal/mol)')
axes[1].set_xlabel('Methyl rotation from the initial staggered geometry (degrees)')
axes[1].set_xticks(np.arange(0,121,15));axes[1].set_xlim(-2,122)
axes[1].axhline(0,color='#424A52',lw=.8)
for ax in axes:
    ax.set_axisbelow(True);ax.grid(color='#E5E8EA',lw=.7)
    ax.axvline(60,color='#A9B0B5',lw=.8,ls=':')
fig.suptitle('Ethane torsion: NNAO and conventional SZ-to-TZ basis sets',fontsize=14,y=.98)
fig.text(.13,.025,'25 rigid geometries, 5-degree spacing; DF-RHF / def2-universal-jkfit, spherical AOs.\n'
    'Each curve uses its own 0-degree energy. HF/cc-pVTZ is a comparison basis, not a complete-basis reference.',
    fontsize=8.4,color='#59636B')
fig.subplots_adjust(left=.13,right=.98,bottom=.12,top=.79)
fig.savefig(OUT/'ethane-torsion-benchmark.png',dpi=200,facecolor='white')
fig.savefig(OUT/'ethane-torsion-benchmark.pdf',facecolor='white')

lines=['# Conventional basis comparison on the ethane torsion scan','',
    'Identical rigid 25-point geometry grid, 0–120 degrees in 5-degree steps, DF-RHF, '
    'def2-universal-jkfit, float64, spherical AOs. NNAO uses frozen update 3351. '
    'Conventional basis coefficients and exponents are fixed at their published library values. '
    'NNAO free contraction is optimized independently for each geometry at fixed exponents and core basis.',
    '', '| Basis | AO count | Grid barrier / kcal mol-1 | Maximum relative-curve difference from cc-pVTZ / kcal mol-1 |',
    '| --- | ---: | ---: | ---: |']
for r in summary:
    lines.append(f"| {r['basis']} | {r['nao']} | {r['grid_barrier_kcal_mol']:.6f} | {r['max_curve_difference_vs_ccpvtz_kcal_mol']:.6f} |")
lines+=['','Each relative curve is referenced to its own 0-degree energy. '
    'HF/cc-pVTZ is the largest conventional basis in this comparison and is not an exact reference. '
    'The barriers refer to this rigid scan, with no nuclear-geometry relaxation.',
    '',f"Traditional basis calculations: GPU4PySCF, job {traditional['job_id']}, one V100, 32 CPU, "
    f"{traditional['elapsed_seconds']:.2f} seconds excluding Python imports. SCF energy tolerance 1e-12 Eh, "
    'orbital gradient tolerance 1e-9, maximum orbital residual < 1e-7. '
    'NNAO energies were independently checked on all 50 frozen endpoint bases.',
    '', 'Command: `sbatch --dependency=afterok:<dense-job-id> artifacts/nnao-ethane-torsion-20261018/traditional.slurm`.',
    'Plots: `ethane-torsion-benchmark.png` and `.pdf`; all numerical curves: `benchmark-curves.csv`.']
(OUT/'BENCHMARK.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
