"""Independent endpoint RHF checks for the methane NNAO experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from pyscf import gto,scf


def validate(path):
    path=Path(path);summary=json.loads(path.read_text());labels=[f'{s}{i}' for i,s in enumerate(summary['symbols'])]
    results={}
    for endpoint in ('initial','final'):
        ecp={label:potential for label,potential in zip(labels,summary.get('ecp') or [None]*len(labels)) if potential}
        mol=gto.M(ecp=ecp,atom=list(zip(labels,summary['coords_angstrom'])),unit='Angstrom',cart=summary['cartesian'],
                  basis=dict(zip(labels,summary[endpoint+'_basis'])),charge=0,spin=0,verbose=0)
        mf=scf.RHF(mol)
        if summary.get('eri_backend')=='ri':mf=mf.density_fit(auxbasis=summary['auxbasis'])
        mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=150;mf.init_guess='1e';mf.kernel()
        assert mf.converged
        energy=float(mf.e_tot);error=abs(energy-summary[endpoint+'_energy_hartree'])
        np.testing.assert_allclose(energy,summary[endpoint+'_energy_hartree'],atol=1e-8,rtol=0)
        results[endpoint]=dict(pyscf_energy_hartree=energy,absolute_error_hartree=error)
    (path.parent/'pyscf-validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('summary',type=Path)
    validate(parser.parse_args().summary)
