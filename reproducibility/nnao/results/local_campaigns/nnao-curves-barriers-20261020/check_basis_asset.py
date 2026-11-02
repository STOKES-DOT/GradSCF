"""Independent same-shell HF check of the exact BSE Br asset; no gradients."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, 'tests/comparisons')
from nnao_reactivity.run import conventional_shells, single_point

root = Path(__file__).resolve().parent
data = json.loads((root/'dataset.json').read_text())
point = next(r for r in data['curves'] if r['group'] == '1.28' and r['scale'] == 1.)
row = next(r for r in point['monomers'] if 'Br' in r['symbols'])
shells = conventional_shells(row, '6-31g(d,p)', data['basis_assets'])
started = time.perf_counter()
native = single_point(row, shells, data['auxbasis'])
labels = [symbol+str(i) for i, symbol in enumerate(row['symbols'])]
mol = gto.M(atom=list(zip(labels, row['coords_angstrom'])),
    basis=dict(zip(labels, shells)), cart=False, charge=row['charge'], spin=row['spin'], verbose=0)
mf = scf.RHF(mol).density_fit(auxbasis=data['auxbasis'])
mf.conv_tol = 1e-12
mf.max_cycle = 200
reference = mf.kernel()
if not mf.converged:
    raise RuntimeError('Independent PySCF reference did not converge.')
np.testing.assert_allclose(native['energy_hartree'], reference, atol=2e-9, rtol=0)
result = dict(status='passed', native=native, reference_hartree=reference,
    difference_hartree=native['energy_hartree']-reference,
    seconds=time.perf_counter()-started, backend='CPU float64', molecule=row,
    basis='6-31G(d,p), spherical AOs; exact BSE Br v1 asset',
    dataset_sha256=hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest(),
    performed_gradients=False, finite_differences=False)
(root/'basis-asset-check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
