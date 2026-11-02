"""CPU same-basis DF-RHF reference when CUDA reference initialization stalls."""
from pathlib import Path
import json
import os
import time
import traceback
import numpy as np
from pyscf import gto,scf,lib
from gpu4_reference_utils import frozen_basis,atomic_json

OUT=Path(__file__).resolve().parent
lib.num_threads(32)
rows={r['structure_id']:r for r in map(json.loads,(OUT/'test-selection.jsonl').read_text().splitlines())}
frozen=[json.loads(s) for s in (OUT/'frozen-bases.jsonl').read_text().splitlines()]
report=dict(status='running',method='DF-RHF',auxbasis='def2-universal-jkfit',device='CPU / PySCF',
    cpu_threads=32,performed_gradients=False,slurm_job_id=os.environ.get('SLURM_JOB_ID'),results=[])
started=time.perf_counter()
for frozen_row in frozen:
    identifier=frozen_row['structure_id'];row=rows[identifier];label=frozen_row['basis'];tick=time.perf_counter()
    print('BEGIN',identifier,label,flush=True)
    try:
        labels,basis=frozen_basis(row,frozen_row)
        mol=gto.M(atom=list(zip(labels,row['coords_angstrom'])),basis=basis,unit='Angstrom',
            charge=row['charge'],spin=row['spin'],cart=False,verbose=0)
        assert mol.nao_nr()==frozen_row['nao']
        mf=scf.RHF(mol).density_fit(auxbasis=report['auxbasis'])
        mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=200;mf.chkfile=None
        density=mf.get_init_guess(mol,key='hcore');mf.kernel(dm0=density)
        if not mf.converged:
            mf.level_shift=.2;mf.kernel(dm0=density)
            seed=mf.make_rdm1();mf.level_shift=0.;mf.kernel(dm0=seed)
        assert mf.converged and mf.level_shift==0.,'Unshifted SCF did not converge'
        d=mf.make_rdm1();s=mf.get_ovlp();h=mf.get_hcore();v=mf.get_veff(mol,d);f=h+v
        residual=float(np.linalg.norm(f@d@s-s@d@f));energy=float(mf.e_tot)
        error=abs(energy-frozen_row['native_energy_hartree'])
        assert residual<1e-7,(residual,identifier)
        assert error<=1e-7,(error,identifier)
        entry=dict(structure_id=identifier,basis=label,status='passed',energy_hartree=energy,
            native_energy_error_hartree=error,orbital_residual=residual,nao=mol.nao_nr(),seconds=time.perf_counter()-tick)
        print('PASSED',identifier,label,energy,error,flush=True)
    except Exception as error:
        entry=dict(structure_id=identifier,basis=label,status='failed',error=str(error),traceback=traceback.format_exc())
        print('FAILED',identifier,label,str(error),flush=True)
    report['results'].append(entry);report['elapsed_seconds']=time.perf_counter()-started
    atomic_json(OUT/'pyscf-check.json',report)
report.update(status='complete',passed=sum(r['status']=='passed' for r in report['results']),
    failed=sum(r['status']!='passed' for r in report['results']))
atomic_json(OUT/'pyscf-check.json',report)
