"""Early CPU PySCF energy checks inside the existing allocation; no gradients."""
from pathlib import Path
import json
import os
import time
import numpy as np
from pyscf import gto,scf,lib

OUT=Path(__file__).resolve().parent
lib.num_threads(2)
report=dict(status='running',backend='CPU PySCF',threads=2,results=[])
started=time.perf_counter()
for angle in (0,60):
    basis=json.loads((OUT/f'angle-{angle:03d}'/'frozen-basis.json').read_text())
    mol=gto.M(atom=list(zip(basis['atom_labels'],basis['coords_angstrom'])),
        basis=dict(zip(basis['atom_labels'],basis['atom_shells'])),unit='Angstrom',
        charge=0,spin=0,cart=False,verbose=0)
    assert mol.nao_nr()==50
    mf=scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
    mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=200;mf.chkfile=None;mf.level_shift=0.
    mf.kernel(dm0=mf.get_init_guess(mol,key='hcore'))
    assert mf.converged
    d=mf.make_rdm1();s=mf.get_ovlp();f=mf.get_hcore()+mf.get_veff(mol,d)
    error=abs(float(mf.e_tot)-basis['native_energy_hartree'])
    residual=float(np.linalg.norm(f@d@s-s@d@f))
    assert error<=1e-7 and residual<=1e-7,(angle,error,residual)
    report['results'].append(dict(angle_deg=angle,energy_hartree=float(mf.e_tot),
        native_energy_error_hartree=error,orbital_residual=residual,status='passed'))
    print(report['results'][-1],flush=True)
report.update(status='complete',passed=2,elapsed_seconds=time.perf_counter()-started)
temporary=OUT/'early-check.json.tmp'
temporary.write_text(json.dumps(report,indent=2)+'\n')
os.replace(temporary,OUT/'early-check.json')
