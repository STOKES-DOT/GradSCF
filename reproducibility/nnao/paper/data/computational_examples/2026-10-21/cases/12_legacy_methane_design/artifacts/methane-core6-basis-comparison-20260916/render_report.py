"""Render the recorded ten-basis comparison; does not rerun calculations."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
report=json.loads((root/'comparison.json').read_text())
reference=json.loads((root/'pyscf-validation.json').read_text())
rows=report['results']
assert len(rows)==10
order=['sto-3g','3-21g','6-31g','6-31g(d)','NNAO-core6-4s4p2d','6-31g(d,p)','def2-svp','cc-pvdz','def2-tzvp','cc-pvtz']
rows=sorted(rows,key=lambda r:order.index(r['basis']))
labels={'sto-3g':'STO-3G','3-21g':'3-21G','6-31g':'6-31G','6-31g(d)':'6-31G(d)',
        'NNAO-core6-4s4p2d':'NNAO core6 + 4s4p2d','6-31g(d,p)':'6-31G(d,p)',
        'def2-svp':'def2-SVP','cc-pvdz':'cc-pVDZ','def2-tzvp':'def2-TZVP','cc-pvtz':'cc-pVTZ'}
colors=['#d96b28' if r['basis'].startswith('NNAO') else '#3f6f94' for r in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(12,6.7),sharey=True,gridspec_kw={'width_ratios':[1,1.12]})
y=np.arange(len(rows))
ax.barh(y,[r['nao'] for r in rows],height=.6,color=colors)
ax.set_yticks(y,[labels[r['basis']] for r in rows]);ax.invert_yaxis()
ax.set_xlim(0,103);ax.set_xlabel('Contracted spherical AO count, N')
ax.set_title('SCF diagonalization: N x N',loc='left',fontweight='bold',pad=16)
for i,r in enumerate(rows):ax.text(r['nao']+1.3,i,f"{r['nao']} x {r['nao']}",va='center',fontsize=10)
ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
gaps=[r['above_cc_pvtz_millihartree'] for r in rows]
bx.barh(y,gaps,height=.6,color=colors)
bx.set_xscale('symlog',linthresh=2);bx.set_xlim(0,1800)
bx.set_xticks([0,2,10,100,1000],[0,2,10,100,1000])
bx.set_xlabel('Energy above cc-pVTZ / mHa (symlog scale)')
bx.set_title('RHF energy difference',loc='left',fontweight='bold',pad=16)
for i,gap in enumerate(gaps):
 bx.annotate(f'{gap:.3f}',(gap,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=10)
bx.grid(axis='x',alpha=.17);bx.set_axisbelow(True)
fig.suptitle('Methane: basis size and energy in GradSCF',x=.04,ha='left',fontsize=18,fontweight='bold',y=.97)
fig.text(.04,.91,'CH4 | C-H = 1.09 Angstrom | RHF | all electron | spherical AOs',fontsize=11,color='#444444')
fig.text(.04,.035,'Orange: coefficients optimized on this same geometry. cc-pVTZ is a finite-basis reference, not the HF limit.',fontsize=10,color='#444444')
fig.subplots_adjust(left=.22,right=.98,top=.82,bottom=.15,wspace=.25)
for ext in ('png','pdf'):fig.savefig(root/f'comparison.{ext}',dpi=200)
maxerror=max(r['absolute_error_hartree'] for r in reference['results'])
lines=['# Methane: ten-basis comparison in GradSCF','',
'All energies were recomputed with GradSCF native integrals and its RHF SCF solver. PySCF was used only as an independent test reference.','',
'## Conditions','',
f"Tetrahedral CH4, C-H = {report['bond_angstrom']} Angstrom; charge/spin = 0/0; 10 electrons; spherical AOs; float64. No ECP. CPU: {report['cpu_model']}; affinity {report['affinity']}; JAX {report['jax_version']}. Total GradSCF benchmark elapsed: {report['elapsed_seconds']:.3f} s (includes compilation; not a steady-state speed benchmark).",'',
'RHF settings: max_cycle=200, energy tolerance 1e-12 Ha, density tolerance 1e-10, orbital tolerance 1e-9, hcore initial guess. The table uses the saved optimized NNAO coefficients from core6-szp442-20260916, not a newly trained network. Its coefficients were optimized on this exact methane geometry; these results do not establish transferability.','',
'## Results','',
'| Basis | C AOs | AOs per H | Total AOs | Fock matrix | Energy / Ha | Above cc-pVTZ / mHa | SCF cycles |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 lines.append(f"| {labels[r['basis']]} | {r['carbon_ao']} | {r['hydrogen_ao']} | {r['nao']} | {r['nao']} x {r['nao']} | {r['energy_hartree']:.12f} | {r['above_cc_pvtz_millihartree']:.6f} | {r['scf_cycles']} |")
lines+=['','## Matrix interpretation','',
'The SCF eigenproblem is F C = S C epsilon. F is the effective one-electron Fock matrix; Hcore and S have the same N x N shape. GradSCF forms an N x N orthogonalized Fock matrix for diagonalization. N is the contracted AO count, including angular components, not the primitive count. Methane has 5 occupied spatial orbitals for every basis; all N orbitals are retained by this solver. No overlap eigenvalues in this experiment approach the clipping threshold.','',
'## Verification and reproducibility','',
f"All ten SCFs converged; every Fock and MO matrix shape was checked, as were FDS-SDF and generalized eigen-equation residuals. Independent PySCF {reference['pyscf_version']} agrees in AO count for every row; maximum energy difference = {maxerror:.6g} Ha (acceptance tolerance 1e-8 Ha).",'',
'A preflight detected and fixed a basis-name parsing defect: heavy-atom polarization in single-part parentheses, e.g. 6-31G(d), was previously ignored. The fix is covered by star-alias equivalence and multi-polarization regression tests. All results here use the corrected parser.','',
'Run gradscf-comparison10.sh in the recorded remote source snapshot (/tmp/gradscf-core6-comparison-20260916). Source hashes, settings, timing, CPU and input-summary hash are retained. See comparison.json, comparison.csv, run.log, tests.log, reference.log and pyscf-validation.json. PNG and PDF figures are generated by render_report.py.','']
(root/'README.md').write_text('\n'.join(lines))
print((root/'README.md').read_text())
