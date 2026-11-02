"""Diagnose reuse of an SCF JK-fitting cache in the CPU MP2 reference."""
from pathlib import Path
import json
import time
from pyscf import gto,scf,mp,df,lib

ROOT=Path(__file__).resolve().parent.parent
lib.num_threads(4)
start=time.perf_counter()
row=next(r for r in json.loads((ROOT/'inputs/structures.json').read_text())
         if r['structure_id']=='S22/02/A')
basis=json.loads((ROOT/'frozen/t14.json').read_text())
shells=next(r['atom_shells'] for r in basis['records'] if r['structure_id']==row['structure_id'])
labels=[s+str(i) for i,s in enumerate(row['symbols'])]
aux=json.loads((ROOT/'inputs/auxiliary-bases.json').read_text())['bases']
mol=gto.M(atom=list(zip(labels,row['coords_angstrom'])),basis=dict(zip(labels,shells)),
          unit='Angstrom',cart=False,charge=0,spin=0,verbose=3,max_memory=12000)
mf=scf.RHF(mol).density_fit(auxbasis=aux['def2-universal-jkfit'])
mf.conv_tol=1e-11;mf.conv_tol_grad=1e-8;mf.max_cycle=200;mf.chkfile=None
mf.kernel();assert mf.converged
old=mp.MP2(mf,frozen=1).density_fit(auxbasis=aux['aug-cc-pvqz-ri'])
old.kernel(with_t2=False)
fresh_df=df.DF(mol,auxbasis=aux['aug-cc-pvqz-ri'])
fresh=mp.MP2(mf,frozen=1).density_fit(auxbasis=aux['aug-cc-pvqz-ri'],with_df=fresh_df)
assert fresh.with_df._cderi is None
fresh.kernel(with_t2=False)
expected=df.addons.make_auxmol(mol,aux['aug-cc-pvqz-ri']).nao_nr()
assert fresh.with_df.get_naoaux()==expected
assert old.with_df._cderi is mf.with_df._cderi
old_cached_rows=int(old.with_df._cderi.shape[0])
assert old_cached_rows!=expected
gpu=next(r for r in json.loads((ROOT/'runtime/water-validation-t14.json').read_text())['results']
         if r['method']=='mp2')['gpu_energy_hartree']
error=abs(float(mf.e_tot+fresh.e_corr)-gpu)
assert error<1e-7,error
assert abs(float(mf.e_tot+old.e_corr)-gpu)>1e-7
report=dict(status='passed',expected_mp2_auxiliary_count=expected,
    old_cached_auxiliary_rows=old_cached_rows,
    old_reported_auxiliary_count=old.with_df.get_naoaux(),fresh_auxiliary_count=fresh.with_df.get_naoaux(),
    old_energy_hartree=float(mf.e_tot+old.e_corr),fresh_energy_hartree=float(mf.e_tot+fresh.e_corr),
    gpu_energy_hartree=gpu,fresh_gpu_error_hartree=error,tolerance_hartree=1e-7,
    elapsed_seconds=time.perf_counter()-start)
(ROOT/'runtime/mp2-reference-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
