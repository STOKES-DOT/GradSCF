"""Conventional SZ-to-TZ DF-RHF curves on the identical rigid torsion grid."""
from pathlib import Path
import csv
import json
import os
import sys
import time
import cupy as cp
import numpy as np
import pyscf
import gpu4pyscf
from pyscf import gto,lib
from gpu4pyscf.scf import hf

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'nnao-relative500-20261018'))
from gpu4_reference_utils import atomic_json

lib.num_threads(32)
rows=json.loads((OUT/'structures-dense.json').read_text())
bases=['sto-3g','3-21g','6-31g(d,p)','cc-pvdz','cc-pvtz']
report=dict(status='running',job_id=os.environ.get('SLURM_JOB_ID'),method='DF-RHF',
    auxbasis='def2-universal-jkfit',dtype='float64',conv_tol=1e-12,conv_tol_grad=1e-9,
    max_cycle=200,cpus=32,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
    versions=dict(pyscf=pyscf.__version__,gpu4pyscf=getattr(gpu4pyscf,'__version__','unavailable'),
                  cupy=cp.__version__,numpy=np.__version__),
    geometry_file='structures-dense.json',results=[])
basis_library={}
started=time.perf_counter()
for basis in bases:
    for row in rows:
        tick=time.perf_counter()
        mol=gto.M(atom=list(zip(row['symbols'],row['coords_angstrom'])),basis=basis,
            unit='Angstrom',charge=0,spin=0,cart=False,verbose=0)
        if basis not in basis_library:
            basis_library[basis]=mol._basis
            atomic_json(OUT/'traditional-basis-library.json',basis_library)
        mf=hf.RHF(mol).density_fit(auxbasis=report['auxbasis'])
        mf.conv_tol=report['conv_tol'];mf.conv_tol_grad=report['conv_tol_grad']
        mf.max_cycle=report['max_cycle'];mf.chkfile=None;mf.level_shift=0.
        density=cp.asarray(mf.get_init_guess(mol,key='hcore'),dtype=cp.float64)
        mf.kernel(dm0=density)
        rescued=False
        if not mf.converged:
            rescued=True
            mf.level_shift=.2;mf.kernel(dm0=density)
            density=mf.make_rdm1();mf.level_shift=0.;mf.kernel(dm0=density)
        assert mf.converged and mf.level_shift==0.,(basis,row['angle_deg'])
        density=mf.make_rdm1();s=mf.get_ovlp();f=mf.get_hcore()+mf.get_veff(mol,density)
        residual=float(cp.linalg.norm(f@density@s-s@density@f).get())
        energy=float(mf.e_tot)
        assert np.isfinite(energy) and residual<1e-7,(basis,row['angle_deg'],residual)
        result=dict(basis=basis,angle_deg=row['angle_deg'],energy_hartree=energy,
            nao=mol.nao_nr(),status='passed',orbital_residual=residual,rescue_used=rescued,
            seconds=time.perf_counter()-tick)
        report['results'].append(result);report['elapsed_seconds']=time.perf_counter()-started
        atomic_json(OUT/'traditional-results.json',report)
        print(result,flush=True)
        del mf,density,s,f
        cp.get_default_memory_pool().free_all_blocks()
assert len(report['results'])==125
report.update(status='complete',passed=len(report['results']),elapsed_seconds=time.perf_counter()-started)
atomic_json(OUT/'traditional-results.json',report)
with (OUT/'traditional-energies.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(report['results'][0]))
    writer.writeheader();writer.writerows(report['results'])
