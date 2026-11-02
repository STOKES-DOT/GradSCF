"""Independent same-frozen-basis DF-RHF check of both curves at every angle."""
from pathlib import Path
import json
import os
import sys
import time
import cupy as cp
from pyscf import gto, lib
from gpu4pyscf.scf import hf

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent/'nnao-relative500-20261018'))
from gpu4_reference_utils import frozen_basis, atomic_json

lib.num_threads(32)
rows = {r['structure_id']:r for r in json.loads((OUT/'structures.json').read_text())}
report = dict(status='running', method='DF-RHF', auxbasis='def2-universal-jkfit',
    job_id=os.environ.get('SLURM_JOB_ID'), results=[])
coarse=json.loads((OUT/'gpu4-check.json').read_text())
assert coarse['status']=='complete' and coarse['passed']==26
atomic_json(OUT/'gpu4-check-coarse.json',coarse)
report['coarse_job_id']=coarse['job_id']
report['results']=list(coarse['results'])
started = time.perf_counter()
for row in rows.values():
    if int(row['angle_deg']) % 10 == 0:
        continue
    folder=OUT/f"angle-{int(row['angle_deg']):03d}"
    for basis in map(json.loads, (folder/'frozen-bases.jsonl').read_text().splitlines()):
        labels, shells = frozen_basis(row, basis)
        mol = gto.M(atom=list(zip(labels, row['coords_angstrom'])), basis=shells,
            unit='Angstrom', charge=row['charge'], spin=row['spin'], cart=False, verbose=0)
        assert mol.nao_nr() == basis['nao']
        mf = hf.RHF(mol).density_fit(auxbasis=report['auxbasis'])
        mf.conv_tol=1e-12; mf.conv_tol_grad=1e-9; mf.max_cycle=200
        mf.chkfile=None; mf.level_shift=0.
        mf.kernel(dm0=cp.asarray(mf.get_init_guess(mol, key='hcore'), dtype=cp.float64))
        assert mf.converged and mf.level_shift == 0.
        d=mf.make_rdm1(); s=mf.get_ovlp(); f=mf.get_hcore()+mf.get_veff(mol,d)
        residual=float(cp.linalg.norm(f@d@s-s@d@f).get())
        energy=float(mf.e_tot); error=abs(energy-basis['native_energy_hartree'])
        assert error <= 1e-7 and residual <= 1e-7, (error,residual)
        report['results'].append(dict(angle_deg=row['angle_deg'], basis=basis['basis'], status='passed',
            energy_hartree=energy, native_energy_error_hartree=error, orbital_residual=residual, nao=mol.nao_nr()))
        atomic_json(OUT/'gpu4-check.json',report)
        print(report['results'][-1],flush=True)
        del mf,d,s,f
        cp.get_default_memory_pool().free_all_blocks()
report.update(status='complete', passed=len(report['results']), elapsed_seconds=time.perf_counter()-started+coarse['elapsed_seconds'])
atomic_json(OUT/'gpu4-check.json',report)
