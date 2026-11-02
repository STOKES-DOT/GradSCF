"""Validate scan results and export numerical comparisons and scientific figures."""
from pathlib import Path
import csv
import json
import os
import numpy as np

OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scan=json.loads((OUT/'scan-summary.json').read_text())
checks=json.loads((OUT/'gpu4-check.json').read_text())
assert scan['status']=='complete' and len(scan['points'])==25
assert checks['status']=='complete' and checks['passed']==50
points=sorted(scan['points'],key=lambda r:r['angle_deg'])
angle=np.array([r['angle_deg'] for r in points])
np.testing.assert_array_equal(angle,np.arange(0,121,5))
assert all(r['nao']==44 for r in points)
assert len({(r['angle_deg'],r['basis']) for r in checks['results']})==50
em=np.array([r['mace_energy_hartree'] for r in points])
ef=np.array([r['free_energy_hartree'] for r in points])
factor=627.5094740631
rm=(em-em[0])*factor
rf=(ef-ef[0])*factor
gap=(em-ef)*factor
shape=rm-rf
assert np.all(em-ef>=-1e-9)
rows=[]
for i,point in enumerate(points):
    row=dict(point, mace_relative_kcal_mol=float(rm[i]), free_relative_kcal_mol=float(rf[i]),
             relative_curve_difference_kcal_mol=float(shape[i]))
    rows.append(row)
with (OUT/'comparison.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)

index60=int(np.flatnonzero(angle==60)[0])
metrics=dict(checkpoint_update=3351, checkpoint_epoch=335, point_count=len(points),
    grid_spacing_degrees=5, method='DF-RHF', basis='nnao_szp_l',
    auxbasis='def2-universal-jkfit', nao=points[0]['nao'],
    source_geometry='GMTKN55-v2/ADIM6/AM2', rigid_scan=True,
    source_geometry_in_training=True,
    mace_grid_barrier_kcal_mol=float(np.ptp(rm)), free_grid_barrier_kcal_mol=float(np.ptp(rf)),
    mace_60_minus_0_kcal_mol=float(rm[index60]), free_60_minus_0_kcal_mol=float(rf[index60]),
    barrier_difference_kcal_mol=float(np.ptp(rm)-np.ptp(rf)),
    absolute_gap_min_kcal_mol=float(gap.min()), absolute_gap_max_kcal_mol=float(gap.max()),
    absolute_gap_mean_kcal_mol=float(gap.mean()),
    relative_curve_max_abs_difference_kcal_mol=float(np.abs(shape).max()),
    relative_curve_rms_difference_kcal_mol=float(np.sqrt(np.mean(shape**2))),
    mace_120_closure_hartree=float(abs(em[-1]-em[0])),
    free_120_closure_hartree=float(abs(ef[-1]-ef[0])),
    mace_mirror_max_error_hartree=float(np.abs(em-em[::-1]).max()),
    free_mirror_max_error_hartree=float(np.abs(ef-ef[::-1]).max()),
    gradient_converged_points=sum(r['coefficient_gradient_converged'] for r in points),
    gradient_converged_starts=sum(r['converged_starts'] for r in points),
    optimization_starts=sum(r['starts'] for r in points),
    maximum_coefficient_gradient_norm=max(r['coefficient_gradient_norm'] for r in points),
    maximum_start_spread_hartree=max(r['endpoint_spread_hartree'] for r in points),
    independent_check_count=checks['passed'],
    maximum_independent_energy_error_hartree=max(r['native_energy_error_hartree'] for r in checks['results']),
    scan_seconds=scan['elapsed_seconds'], independent_check_seconds=checks['elapsed_seconds'])
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
    'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#656B70',
    'text.color':'#24292F','axes.labelcolor':'#24292F'})
fig,axes=plt.subplots(2,1,figsize=(8.4,7.1),sharex=True,
    gridspec_kw={'height_ratios':[1.65,1],'hspace':.20})
axes[0].plot(angle,rm,'o-',color='#D87632',lw=2,ms=5,label='MACE / update 3351')
axes[0].plot(angle,rf,'s--',color='#28699C',lw=2,ms=4.5,label='Free contraction / same pool')
axes[0].set_ylabel(r'$E(\phi)-E(0^\circ)$ (kcal/mol)')
axes[0].legend(frameon=False,loc='best')
axes[0].set_title('Ethane: rigid rotation of one methyl group',pad=13,fontsize=14)
axes[1].plot(angle,gap,'o-',color='#4B5660',lw=1.8,ms=4)
axes[1].axhline(gap[0],color='#89939B',ls=':',lw=1,label='Gap at 0 degrees')
axes[1].set_ylabel(r'$E_{\mathrm{MACE}}-E_{\mathrm{free}}$'+'\n(kcal/mol)')
axes[1].set_xlabel('Methyl rotation from the initial staggered geometry (degrees)')
axes[1].set_xticks(np.arange(0,121,20));axes[1].set_xlim(-2,122)
for ax in axes:
    ax.set_axisbelow(True);ax.grid(color='#E4E7E9',lw=.7)
    ax.axvline(60,color='#A9B0B5',lw=.9,ls=':')
fig.text(.12,.025,'DF-RHF; 44 spherical AOs; fixed exponents and core basis. Independent coefficient optimization at each angle.\n'
    'Upper curves each use their own 0-degree energy; lines connect the computed 5-degree grid points.',
    fontsize=8.2,color='#59636B')
fig.subplots_adjust(left=.15,right=.98,bottom=.14,top=.91)
fig.savefig(OUT/'ethane-torsion.png',dpi=200,facecolor='white')
fig.savefig(OUT/'ethane-torsion.pdf',facecolor='white')

lines=['# Ethane torsion: MACE versus free contraction','',
    'Rigid 0–120 degree methyl rotation in 5 degree steps, starting from training geometry '
    '`GMTKN55-v2/ADIM6/AM2`; bond lengths and valence angles are fixed.',
    'MACE uses the frozen update-3351 checkpoint. Exponents, 6-31G core contractions, '
    'single-zeta plus polarization topology and all-electron DF-RHF settings are identical.',
    'All angles have 44 spherical AOs. Each free-contraction problem has independent per-atom '
    'radial coefficients, shared by the magnetic components of each shell; core coefficients stay fixed.',
    'Two starts (MACE and reference coefficients), analytic implicit SCF gradients, L-BFGS-B, '
    '1000-iteration cap; coefficient gradient norm <= 1e-5 is reported as gradient-converged. '
    'Finite-difference validation is not performed.',
    'SCF tolerances: energy 1e-12 Eh, density 1e-10, orbital gradient 1e-9; implicit tolerance 1e-8. '
    'Native packed DF with def2-universal-jkfit; no full four-index ERI.',
    '', '| Quantity | Value |','| --- | ---: |']
for key in ['mace_grid_barrier_kcal_mol','free_grid_barrier_kcal_mol','barrier_difference_kcal_mol',
            'absolute_gap_min_kcal_mol','absolute_gap_max_kcal_mol',
            'relative_curve_max_abs_difference_kcal_mol','relative_curve_rms_difference_kcal_mol',
            'gradient_converged_points','gradient_converged_starts','maximum_start_spread_hartree',
            'maximum_independent_energy_error_hartree','scan_seconds','independent_check_seconds']:
    lines.append(f'| {key} | {metrics[key]:.10g} |')
lines+=['','Run on jy_gpu_2, Slurm job '+str(scan['job_id'])+
    ', following coarse-grid job '+str(scan['coarse_job_id'])+
    '; one V100, 32 CPU, 128 GiB requested RAM per job, run sequentially. '
    'Commands: `sbatch artifacts/nnao-ethane-torsion-20261018/run.slurm`, then '
    '`sbatch --dependency=afterok:<coarse-job-id> artifacts/nnao-ethane-torsion-20261018/dense.slurm`.',
    'The 0–120 degree closure and mirror errors are retained in `metrics.json`. '
    'All 50 frozen bases are checked independently with GPU4PySCF. '
    'Finite-budget agreement between starts does not prove a global coefficient minimum. '
    'Grid barriers are measured on this rigid scan, not relaxed-geometry or complete-basis barriers.',
    '', 'Raw per-angle energies: `comparison.csv`; optimization histories and coefficients: `angle-*/`; '
    'full geometry sequence: `torsion-dense.xyz`.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(metrics,indent=2))
