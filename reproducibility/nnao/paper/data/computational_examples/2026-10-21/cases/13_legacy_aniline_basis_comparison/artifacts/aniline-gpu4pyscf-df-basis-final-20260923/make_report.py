from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
report=json.loads((root/'comparison.json').read_text())
rows=[row for row in report['results'] if row['status']=='pass']
nnao=next(row for row in rows if row['basis'].startswith('NNAO'))

fig,axes=plt.subplots(1,2,figsize=(14,7),constrained_layout=True)
ax=axes[0]
offsets={
    '6-31g':(5,-13),'NNAO-szp663-trained':(5,7),'6-31g(d)':(5,5),
    '6-31g(d,p)':(5,12),'def2-svp':(5,17),'cc-pvdz':(5,-13),
}
for row in rows:
    custom=row is nnao
    ax.scatter(row['nao'],row['above_cc_pvtz_millihartree'],s=150 if custom else 55,
               marker='*' if custom else 'o',color='#c62828' if custom else '#1565c0',zorder=3)
    label='NNAO trained' if custom else row['basis']
    ax.annotate(label,(row['nao'],row['above_cc_pvtz_millihartree']),
                xytext=offsets.get(row['basis'],(5,5)),textcoords='offset points',fontsize=8)
ax.axhline(0,color='black',linewidth=.8)
ax.set_yscale('symlog',linthresh=10)
ax.set_xlabel('Contracted AO count / Hamiltonian dimension')
ax.set_ylabel(r'$E-E_{\mathrm{cc-pVTZ}}$ / m$E_h$')
ax.set_title('Finite-basis RHF energy versus AO size')
ax.grid(alpha=.25)

ax=axes[1]
conventional=sorted((row for row in rows if row is not nnao),key=lambda row:row['energy_hartree'],reverse=True)
delta=np.asarray([1000*(row['energy_hartree']-nnao['energy_hartree']) for row in conventional])
colors=np.where(delta>0,'#ef6c00','#1565c0')
y=np.arange(len(conventional))
ax.barh(y,delta,color=colors)
ax.set_yticks(y,[row['basis'] for row in conventional],fontsize=9)
ax.invert_yaxis();ax.axvline(0,color='#c62828',linewidth=1.5,label='trained NNAO')
ax.set_xscale('symlog',linthresh=50)
ax.set_xlabel(r'$E_{\mathrm{basis}}-E_{\mathrm{NNAO}}$ / m$E_h$')
ax.set_title('Energy relative to trained NNAO')
ax.grid(axis='x',alpha=.25);ax.legend()

fig.suptitle('Aniline GPU4PySCF DF-RHF, spherical AOs, cc-pVQZ-JKFIT',fontsize=14)
fig.savefig(root/'basis_comparison.png',dpi=220)
fig.savefig(root/'basis_comparison.pdf')

lines=['# Aniline GPU4PySCF basis comparison','',
       'All rows use the same fixed aniline geometry, spherical AOs, RHF, float64, and '
       '`cc-pVQZ-JKFIT`. The trained NNAO basis is evaluated independently by GPU4PySCF; '
       'the training energy is not reused.','',
       '| Basis | Level | AO / H dimension | Energy / Ha | vs cc-pVTZ / mHa | SCF cycles | Time / s |',
       '|---|---|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['basis']} | {row['level']} | {row['nao']} | {row['energy_hartree']:.12f} | "
                 f"{row['above_cc_pvtz_millihartree']:.3f} | {row['scf_cycles']} | {row['scf_seconds']:.2f} |")
nearest=min((row for row in rows if row is not nnao),key=lambda row:abs(row['energy_hartree']-nnao['energy_hartree']))
lines += ['', '## Interpretation','',
          f"- The trained NNAO has **{nnao['nao']} AOs**, between unpolarized 6-31G/3-21G "
          '(77 AOs) and 6-31G(d) (112 AOs).',
          f"- Its nearest conventional energy is **{nearest['basis']}**: NNAO is "
          f"{1000*(nearest['energy_hartree']-nnao['energy_hartree']):.3f} mHa lower.",
          f"- NNAO remains {1000*(nnao['energy_hartree']-next(r['energy_hartree'] for r in rows if r['basis']=='6-31g')):.3f} mHa "
          'above 6-31G and does not reach polarized double-zeta or triple-zeta energies.',
          '- This is a single-geometry variational comparison. It does not establish transferability, forces, or basis-set-limit accuracy.','']
(root/'README.md').write_text('\n'.join(lines))
