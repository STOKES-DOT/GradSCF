"""Append the measured expanded-basis row while preserving the previous ten."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
baseline=root.parent/'methane-core6-basis-comparison-20260916'
newpath=root/'artifacts/methane-szp663'
report=json.loads((baseline/'comparison.json').read_text())
new=json.loads((newpath/'comparison/comparison.json').read_text())
assert len(report['results'])==10 and len(new['results'])==1
for key in ['method','cartesian','symbols','charge','spin','dtype','settings']:
 assert report[key]==new[key],key
np.testing.assert_array_equal(report['coords_angstrom'],new['coords_angstrom'])
row=new['results'][0]
assert row['basis']=='NNAO-core6-6s6p3d-H6s2p' and row['nao']==26 and row['primitive_nao']==93
anchor=next(r['energy_hartree'] for r in report['results'] if r['basis']=='cc-pvtz')
row['above_cc_pvtz_millihartree']=(row['energy_hartree']-anchor)*1000
report['results'].append(row)
report['appended_run']={k:v for k,v in new.items() if k!='results'}
report['baseline_comparison_sha256']=hashlib.sha256((baseline/'comparison.json').read_bytes()).hexdigest()
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/'comparison.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(report['results'][0]));writer.writeheader();writer.writerows(report['results'])
ref=json.loads((baseline/'pyscf-validation.json').read_text())
newref=json.loads((newpath/'comparison/pyscf-validation.json').read_text())
assert ref['pyscf_version']==newref['pyscf_version']
ref['results']+=newref['results']
(root/'pyscf-validation.json').write_text(json.dumps(ref,indent=2)+'\n')
s=json.loads((newpath/'summary.json').read_text())
old=next(r for r in report['results'] if r['basis']=='NNAO-core6-4s4p2d')
assert abs(s['initial_energy_hartree']-old['energy_hartree'])<1e-9
assert row['energy_hartree']<=old['energy_hartree']+1e-9
text=f'''# Expanded primitive optimization

C: fixed six-primitive core, valence/polarization 4s4p2d -> 6s6p3d.
H: 3s1p -> 6s2p; the p contraction is now trainable. No added contracted
radial functions; CH4 stays at 26 spherical AOs, while primitive AOs grow
from 56 to 93. Primitive ERI storage grows by (93/56)^4 = {(93/56)**4:.3f}.

Old exponents are preserved, with added tight/diffuse endpoints (2*max,
min/2), then a geometric midpoint in the largest logarithmic gap where
needed. All exponents stay fixed. Core coefficients stay fixed; MACE
predicts complete valence/polarization vectors, including new H-p vectors.
The output head is (natom,3,6). This is a custom unoptimized exponent grid.

Initialization embeds the preceding optimized coefficients with new entries
zero. The trainable output bias is initialized; the MACE backbone uses seed
0 and zero head kernels. This transfers coefficients, not a whole trained
network checkpoint. No fixed empirical vector is added in forward.

- Initial RHF: {s['initial_energy_hartree']:.12f} Ha.
- Final RHF: {s['final_energy_hartree']:.12f} Ha.
- Improvement relative to preceding optimized basis: {(old['energy_hartree']-row['energy_hartree'])*1000:.6f} mHa.
- Accepted steps: {s['iterations']}; objective evaluations: {s['evaluations']}.
- Optimizer success: {s['optimizer_success']}; termination: {s['message']}.
- Maximum final parameter gradient: {s['final']['max_parameter_gradient']:.6g} (gtol=5e-7).
- Network direction FD error, initial/final: {s['initial_fd']['absolute_error']:.6g} / {s['final_fd']['absolute_error']:.6g}, step=1e-4.
- Training elapsed: {s['elapsed_seconds']:.3f} s on CPU; hardware is recorded in the independent comparison JSON.

Local regression: 82 passed, 6 optional dependency skips. Remote NNAO suite:
80 passed (143.88 s), including actual MACE/JIT. All 710 recorded source and
resource hashes match local files. The wheel contains the expanded basis
and head code. The initial 160-step trial is preserved in
artifacts/methane-szp663-160steps/; the repeated first 160 energies match
exactly in the final run.

The derivative test uses the stationary first-order RHF Lagrangian with
Pulay terms and fixed native primitive integrals. Higher-order derivatives
are not tested here. Neither optimized basis has been validated for transfer
to other molecules/geometries. All benchmark results remain finite-basis RHF.
'''
(root/'optimization-details.md').write_text(text)
print(text)
