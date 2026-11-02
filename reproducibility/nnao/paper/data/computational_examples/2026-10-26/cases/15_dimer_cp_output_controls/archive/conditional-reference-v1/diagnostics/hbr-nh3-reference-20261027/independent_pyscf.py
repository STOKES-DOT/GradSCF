"""Independent same-basis HF/DF check of the anomalous dimer's five energies."""
from pathlib import Path
import json
import time

import numpy as np
from pyscf import df,gto,lib,scf


HERE=Path(__file__).resolve().parent
CASE=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper/data/computational_examples/2026-10-26/cases/15_dimer_cp_output_controls')
ASSETS=CASE.parent/'14_bsse_counterpoise/frozen-bases/distance'
lib.num_threads(1)


def molecule(row,shells,kept,*,ghost):
    indices=list(range(len(shells))) if ghost else kept
    labels=[(row['symbols'][i]+str(i)) if i in kept else 'Ghost-'+row['symbols'][i]+str(i) for i in indices]
    mol=gto.M(atom=[(label,row['coords_angstrom'][i]) for label,i in zip(labels,indices)],
              basis={label:shells[i] for label,i in zip(labels,indices)},
              charge=0,spin=0,cart=False,verbose=0,max_memory=4000)
    assert mol.nelectron==sum(row['atomic_numbers'][i] for i in kept)
    return mol


def main():
    inputs=json.loads((CASE/'inputs/cases.json').read_text())
    records=json.loads((CASE/'combined/results.json').read_text())['results']
    report=dict(status='running',method='PySCF 2.9.0 DF-RHF with explicit common dimer auxiliary factors',
                local_cpu_threads=1,finite_differences=False,results=[])
    cases=[('NNAO/MACE',.8),('NNAO/MACE-fixed',.8),('NNAO/MACE',1.)]
    for basis,scale in cases:
        native=next(r for r in records if r['basis']==basis and r['group']=='1.28' and r['scale']==scale)
        point=next(p for p in inputs['definitions'] if p['identifier']==native['identifier'])
        if basis=='NNAO/MACE-fixed':
            job=next(j for j in inputs['jobs'] if j['key']==native['key']);row=job['row'];shells=job['atom_shells']
        else:
            saved=json.loads((ASSETS/(native['key']+'-basis.json')).read_text());row=saved['row'];shells=saved['atom_shells']
        all_atoms=list(range(len(shells)))
        full=molecule(row,shells,all_atoms,ghost=False)
        aux=df.addons.make_auxmol(full,'def2-universal-jkfit')
        factors=df.incore.cholesky_eri(full,auxmol=aux,max_memory=4000,verbose=0)
        ranges=full.aoslice_by_atom()[:,2:]
        problems=[('dimer',full,factors,native['dimer_energy_hartree'])]
        for n,kept in enumerate(point['fragment_indices']):
            own=molecule(row,shells,kept,ghost=False)
            ghost=molecule(row,shells,kept,ghost=True)
            np.testing.assert_allclose(ghost.intor_symmetric('int1e_ovlp'),
                full.intor_symmetric('int1e_ovlp'),atol=1e-13,rtol=0)
            ao=np.concatenate([np.arange(*ranges[i]) for i in kept])
            left,right=np.tril_indices(len(ao));a,b=ao[left],ao[right]
            pair=np.maximum(a,b)*(np.maximum(a,b)+1)//2+np.minimum(a,b)
            own_factors=np.ascontiguousarray(factors[:,pair])
            problems += [(f'own/{n}',own,own_factors,native['own_energies_hartree'][n]),
                         (f'ghost/{n}',ghost,factors,native['ghost_energies_hartree'][n])]
        for label,mol,eri,expected in problems:
            start=time.perf_counter();energies=[];starts=[];converged_state=None
            for guess in ('minao','1e'):
                mf=scf.RHF(mol).density_fit();mf.verbose=0
                # Explicit shared factors; DF.build() must not regenerate a
                # different own-fragment auxiliary metric.
                mf.with_df._cderi=eri;mf.with_df.auxmol=None
                mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=200
                mf.init_guess=guess
                value=mf.kernel()
                assert mf._eri is None,'No complete AO ERI may be retained.'
                starts.append(dict(guess=guess,converged=bool(mf.converged),energy_hartree=float(value)))
                if not mf.converged:
                    # An unconverged core-only guess is not an energy result.
                    # Retain that outcome rather than treating it as a second
                    # stationary solution or silently replacing its value.
                    continue
                np.testing.assert_allclose(value,expected,atol=2e-8,rtol=0)
                energies.append(float(value))
                converged_state=mf
            assert starts[0]['converged'] and converged_state is not None
            _,_,stable,_=converged_state.stability(internal=True,external=False,return_status=True,tol=1e-7)
            assert stable,(basis,scale,label)
            result=dict(basis=basis,scale=scale,stage=label,nao=mol.nao_nr(),
                native_energy_hartree=expected,independent_energies_hartree=energies,
                maximum_error_hartree=max(abs(e-expected) for e in energies),
                internal_stability=True,starts=starts,
                seconds=time.perf_counter()-start,status='passed')
            report['results'].append(result)
            (HERE/'independent-check.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    assert len(report['results'])==15
    report.update(status='passed',energy_problems=15,scf_starts=30,
                  unconverged_starts=sum(not s['converged'] for r in report['results'] for s in r['starts']),
                  max_error_hartree=max(r['maximum_error_hartree'] for r in report['results']))
    (HERE/'independent-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Independent 15-energy reproduction passed; individual guess outcomes are retained.',flush=True)


if __name__=='__main__':main()
