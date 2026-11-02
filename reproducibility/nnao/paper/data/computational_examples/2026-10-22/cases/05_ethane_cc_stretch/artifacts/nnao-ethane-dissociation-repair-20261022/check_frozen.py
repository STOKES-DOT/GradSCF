"""Independent restricted-HF branch audit at exactly the archived NNAO bases."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
from pathlib import Path
import json
import time
import numpy as np
from pyscf import gto, scf

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'nnao-ethane-dissociation-20261019'


def run():
    rows = json.loads((SOURCE / 'structures.json').read_text())
    report = dict(status='running', method='DF-RHF', auxbasis='def2-universal-jkfit', results=[])
    previous = {}
    started = time.perf_counter()
    for index, row in enumerate(rows):
        if not 2.55 <= row['distance_angstrom'] <= 3.1:
            continue
        records = [json.loads(s) for s in (SOURCE / f'distance-{index:03d}/frozen-bases.jsonl').read_text().splitlines()]
        for frozen in records:
            name = frozen['basis']
            basis = dict(zip(frozen['atom_labels'], frozen['atom_shells']))
            mol = gto.M(atom=list(zip(frozen['atom_labels'], row['coords_angstrom'])), basis=basis,
                        unit='Angstrom', charge=0, spin=0, cart=False, verbose=0)
            starts = [('hcore', None), ('minao', None)]
            if name in previous:
                pmol, density = previous[name]
                starts.append(('projected_previous', scf.addons.project_dm_nr2nr(pmol, density, mol)))
            results, states = [], []
            for start, density in starts:
                mf = scf.RHF(mol).density_fit(auxbasis=report['auxbasis'])
                mf.conv_tol = 1e-12
                mf.conv_tol_grad = 1e-9
                mf.max_cycle = 200
                mf.chkfile = None
                mf.kernel(dm0=density if density is not None else mf.get_init_guess(mol, key=start))
                if not mf.converged:
                    mf = mf.newton()
                    mf.kernel()
                stable = False
                restarts = 0
                for attempt in range(5):
                    if not mf.converged:
                        break
                    internal, _, stable, _ = mf.stability(external=False, return_status=True)
                    if stable or attempt == 4:
                        break
                    restarts += 1
                    mf.kernel(dm0=mf.make_rdm1(internal, mf.mo_occ))
                density = mf.make_rdm1()
                overlap = mf.get_ovlp()
                fock = mf.get_hcore() + mf.get_veff(mol, density)
                residual = float(np.linalg.norm(fock @ density @ overlap - overlap @ density @ fock))
                valid = bool(mf.converged and stable and residual < 1e-7)
                result = dict(start=start, energy_hartree=float(mf.e_tot), converged=bool(mf.converged),
                    internally_stable=bool(stable), valid=valid, orbital_residual=residual,
                    stability_restarts=restarts)
                results.append(result)
                if valid:
                    states.append((float(mf.e_tot), mf, start))
            if not states:
                raise RuntimeError(f'No independently valid solution: {index}, {name}')
            energy, selected, seed = min(states, key=lambda item: item[0])
            previous[name] = mol, selected.make_rdm1()
            record = dict(index=index, distance_angstrom=row['distance_angstrom'], basis=name,
                archived_energy_hartree=frozen['native_energy_hartree'], best_energy_hartree=energy,
                archived_minus_best_hartree=frozen['native_energy_hartree'] - energy,
                selected_start=seed, starts=results)
            np.savez(OUT / f'branch-{index:03d}-{name}.npz', density=selected.make_rdm1(),
                     mo_coeff=selected.mo_coeff, mo_energy=selected.mo_energy, mo_occ=selected.mo_occ)
            report['results'].append(record)
            report['seconds'] = time.perf_counter() - started
            (OUT / 'frozen-branch-audit.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(record), flush=True)
    report['status'] = 'complete'
    (OUT / 'frozen-branch-audit.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    run()
