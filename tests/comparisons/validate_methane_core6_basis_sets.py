"""Independent PySCF check of all ten GradSCF methane benchmark rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyscf
from pyscf import gto,scf


def validate(comparison,summary):
    comparison=Path(comparison);report=json.loads(comparison.read_text())
    trained=json.loads(Path(summary).read_text())
    labels=[f'{s}{i}' for i,s in enumerate(report['symbols'])]
    rows=[]
    for row in report['results']:
        basis=dict(zip(labels,trained['final_basis'])) if row['basis'].startswith('NNAO-') else row['basis']
        mol=gto.M(atom=list(zip(labels,report['coords_angstrom'])),unit='Angstrom',basis=basis,
                  cart=False,charge=0,spin=0,verbose=0)
        mf=scf.RHF(mol);mf.init_guess='1e';mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=200
        energy=float(mf.kernel())
        assert mf.converged and mol.nao_nr()==row['nao']
        np.testing.assert_allclose(energy,row['energy_hartree'],atol=1e-8,rtol=0)
        entry=dict(basis=row['basis'],nao=mol.nao_nr(),pyscf_energy_hartree=energy,
                   absolute_error_hartree=abs(energy-row['energy_hartree']))
        rows.append(entry);print(entry,flush=True)
    (comparison.parent/'pyscf-validation.json').write_text(json.dumps(dict(pyscf_version=pyscf.__version__,results=rows),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison');parser.add_argument('summary')
    args=parser.parse_args();validate(args.comparison,args.summary)
