"""Independent PySCF check of the saved GradSCF carbon FCI labels; no training."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from pyscf import gto, scf, fci
import pyscf

folder = Path('artifacts/iks-carbon-sto3g-20261026')
saved = np.load(folder/'reference.npz')
report = json.loads((folder/'summary.json').read_text())
started = perf_counter()
mol = gto.M(atom='C 0 0 0', basis='sto-3g', charge=0, spin=2, cart=True, verbose=0)
mf = scf.ROHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-11)
solver = fci.FCI(mf)
energies, coefficients = solver.kernel(nroots=3, tol=1e-13)
if not mf.converged or not np.all(solver.converged):
    raise RuntimeError('Independent reference did not converge')
densities = []
for ci in coefficients:
    da, db = solver.make_rdm1s(ci, mol.nao, mol.nelec)
    densities.append(mf.mo_coeff@np.stack((da, db))@mf.mo_coeff.T)
density = np.mean(densities, axis=0)
energy_error = float(np.max(np.abs(np.asarray(energies)-report['fci_total_energy_hartree'])))
density_error = float(np.max(np.abs(density-saved['density_ao'])))
np.testing.assert_allclose(energies, report['fci_total_energy_hartree'], atol=3e-10, rtol=0)
np.testing.assert_allclose(density, saved['density_ao'], atol=1e-8, rtol=0)
result = dict(passed=True, molecule='C', spin=2, basis='sto-3g', cart=True,
    density_convention='equal mixture of the three lowest Ms=1 triplet spatial roots',
    pyscf=pyscf.__version__, energy_error_hartree=energy_error,
    density_max_error_ao=density_error, seconds=perf_counter()-started)
(folder/'pyscf_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
