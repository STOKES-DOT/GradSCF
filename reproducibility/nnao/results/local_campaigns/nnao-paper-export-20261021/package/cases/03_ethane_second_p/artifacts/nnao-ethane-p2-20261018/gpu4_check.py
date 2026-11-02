"""Independent DF-RHF check of every exported 50-AO optimized basis."""
from pathlib import Path
import json
import os
import sys
import time
import cupy as cp
from pyscf import gto,lib
from gpu4pyscf.scf import hf

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'nnao-ethane-torsion-20261018'
sys.path.insert(0,str(OUT.parent/'nnao-relative500-20261018'))
from gpu4_reference_utils import frozen_basis,atomic_json

lib.num_threads(32)
rows=json.loads((BASE/'structures.json').read_text())
report=dict(status='running',job_id=os.environ.get('SLURM_JOB_ID'),method='DF-RHF',
    auxbasis='def2-universal-jkfit',results=[])
started=time.perf_counter()
for row in rows:
    folder=OUT/f"angle-{int(row['angle_deg']):03d}"
    basis=json.loads((folder/'frozen-basis.json').read_text())
    labels,shells=frozen_basis(row,basis)
    mol=gto.M(atom=list(zip(labels,row['coords_angstrom'])),basis=shells,
        unit='Angstrom',charge=0,spin=0,cart=False,verbose=0)
    assert mol.nao_nr()==50
    mf=hf.RHF(mol).density_fit(auxbasis=report['auxbasis'])
    mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=200
    mf.chkfile=None;mf.level_shift=0.
    initial=cp.asarray(mf.get_init_guess(mol,key='hcore'),dtype=cp.float64)
    mf.kernel(dm0=initial)
    if not mf.converged:
        mf.level_shift=.2;mf.kernel(dm0=initial)
        density=mf.make_rdm1();mf.level_shift=0.;mf.kernel(dm0=density)
    assert mf.converged and mf.level_shift==0.
    d=mf.make_rdm1();s=mf.get_ovlp();f=mf.get_hcore()+mf.get_veff(mol,d)
    residual=float(cp.linalg.norm(f@d@s-s@d@f).get())
    energy=float(mf.e_tot);error=abs(energy-basis['native_energy_hartree'])
    assert error<=1e-7 and residual<=1e-7,(row['angle_deg'],error,residual)
    report['results'].append(dict(angle_deg=row['angle_deg'],status='passed',nao=50,
        energy_hartree=energy,native_energy_error_hartree=error,orbital_residual=residual))
    atomic_json(OUT/'gpu4-check.json',report)
    print(report['results'][-1],flush=True)
    del mf,initial,d,s,f
    cp.get_default_memory_pool().free_all_blocks()
report.update(status='complete',passed=len(report['results']),elapsed_seconds=time.perf_counter()-started)
atomic_json(OUT/'gpu4-check.json',report)
