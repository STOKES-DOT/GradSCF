"""RHF basis comparison at the exact methane geometry used for NNAO training.

All rows are independently reconverged in the same PySCF environment. The
larger def2-TZVP result is a finite-basis reference, not a CBS or exact energy.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import pyscf
from pyscf import gto,scf,lib


def compare(summary_path,output):
    source=Path(summary_path);trained=json.loads(source.read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    labels=[f'{s}{i}' for i,s in enumerate(trained['symbols'])]
    atom=list(zip(labels,trained['coords_angstrom']))
    cases=[(name,name) for name in ['sto-3g','3-21g','6-31g','6-31g(d)','6-31g(d,p)','def2-svp','cc-pvdz','def2-tzvp']]
    # Controlled baseline: retain 3-21G split valence and add the same fixed
    # C d / H p polarization shells as NNAO, with the same primitive pool.
    matched={label:copy.deepcopy(gto.basis.load('3-21g',symbol))+[copy.deepcopy(shells[-1])]
             for label,symbol,shells in zip(labels,trained['symbols'],trained['initial_basis'])}
    cases.append(('3-21g+same-P',matched))
    for endpoint in ['initial','final']:
        cases.append(('NNAO-'+endpoint,dict(zip(labels,trained[endpoint+'_basis']))))
    results=[];started=time.perf_counter()
    for name,basis in cases:
        begin=time.perf_counter()
        mol=gto.M(atom=atom,unit='Angstrom',basis=basis,cart=True,charge=0,spin=0,verbose=0)
        mf=scf.RHF(mol);mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=200;mf.init_guess='1e'
        energy=float(mf.kernel())
        residual=float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ)))
        if not mf.converged or not np.isfinite(energy) or residual>1e-7:
            raise RuntimeError(f'{name} failed: converged={mf.converged}, gradient={residual}')
        nao=mol.nao_nr()
        nprim=sum(mol.bas_nprim(i)*(mol.bas_angular(i)+1)*(mol.bas_angular(i)+2)//2 for i in range(mol.nbas))
        row=dict(basis=name,nao=nao,primitive_nao=nprim,energy_hartree=energy,
                 converged=bool(mf.converged),orbital_gradient_norm=residual,
                 elapsed_seconds=time.perf_counter()-begin)
        if name.startswith('NNAO-'):
            target=trained[name.split('-')[1]+'_energy_hartree']
            np.testing.assert_allclose(energy,target,atol=1e-8,rtol=0)
        results.append(row)
        print(f'{name:18s} AO={nao:3d} primitiveAO={nprim:3d} E={energy:.12f} |g|={residual:.2e}',flush=True)
    anchor=next(r['energy_hartree'] for r in results if r['basis']=='def2-tzvp')
    for row in results:row['above_def2_tzvp_millihartree']=(row['energy_hartree']-anchor)*1000
    report=dict(method='RHF',geometry_source=str(source),geometry_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        atom_angstrom=atom,bond_angstrom=trained['bond_angstrom'],cartesian=True,charge=0,spin=0,
        settings=dict(conv_tol=1e-12,conv_tol_grad=1e-9,max_cycle=200,init_guess='1e'),
        energy_reference='def2-tzvp (finite basis; not CBS)',pyscf_version=pyscf.__version__,
        numpy_version=np.__version__,python=platform.python_version(),platform=platform.platform(),
        cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown') if Path('/proc/cpuinfo').exists() else platform.processor(),
        threads=lib.num_threads(),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started,results=results)
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print('Elapsed / s:',report['elapsed_seconds'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('--output',type=Path,default=Path('artifacts/methane-basis-comparison'))
    args=parser.parse_args();compare(args.summary,args.output)
