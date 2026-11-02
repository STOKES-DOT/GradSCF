"""Independent fixed-basis validation of repaired cases, never a solver bridge."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, 'tests/comparisons')
from gradscf.model.nnao.runtime import atomic_json, file_hash

PRIORITY = ['fb81f9654e02577e0508', 'f8ce7859d72f242c10e1', 'ea90966f585e090d1442',
            '20c7cb8f37f742ca8e54', 'a51aa9ee02b4857e5efb', '791331ccd7a3b1428add']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--retry', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--extra-keys-file', type=Path)
    args = parser.parse_args()
    repaired = json.loads((args.retry/'results.json').read_text())
    selection = json.loads(args.selection.read_text())
    extras = set(json.loads(args.extra_keys_file.read_text())) if args.extra_keys_file else set()
    selected = set(selection['original_failures']) | set(PRIORITY) | extras
    rows = [r for r in repaired['results'] if r['key'] in selected]
    if {r['key'] for r in rows} != selected:
        raise ValueError('Independent check is missing a requested case.')
    report = dict(status='running', requested_cases=len(rows), passed=0, failed=0,
                  retry_sha256=file_hash(args.retry/'results.json'),
                  scope='same exported orbital basis, DF-HF energy/stationarity; selected independent internal stability', results=[])
    for row in rows:
        tick = time.perf_counter()
        record = dict(key=row['key'], identifier=row['identifier'], basis=row['basis'])
        try:
            if row['status'] != 'passed':
                raise RuntimeError('Native repair did not pass: '+row.get('error', row['status']))
            exported = json.loads((args.parent/f'{row["key"]}-basis.json').read_text())
            r, shells = exported['row'], exported['atom_shells']
            suffix = '-state.npz' if row['task'] == 'species' else '-dimer-state.npz'
            with np.load(args.retry/(row['key']+suffix)) as saved:
                ds = saved['density_spin']
            labels = [symbol+str(i) for i, symbol in enumerate(r['symbols'])]
            mol = gto.M(atom=list(zip(labels, r['coords_angstrom'])), basis=dict(zip(labels, shells)),
                        charge=r['charge'], spin=r['spin'], cart=False, verbose=0)
            mf = (scf.UHF(mol) if mol.spin else scf.RHF(mol)).density_fit(auxbasis='def2-universal-jkfit')
            mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-8, 100
            expected = row['energy_hartree'] if row['task'] == 'species' else row['dimer_energy_hartree']
            energy = mf.kernel(dm0=ds if mol.spin else ds.sum(axis=0))
            if not mf.converged:
                raise RuntimeError('Independent same-basis reference did not converge from the retained native density.')
            np.testing.assert_allclose(energy, expected, atol=2e-8, rtol=0)
            record.update(native_energy=expected, reference_energy=float(energy),
                difference_hartree=float(energy-expected), reference_converged=True,
                reference_spin_squared=float(mf.spin_square()[0]) if mol.spin else 0.,
                four_index_eri_retained=mf._eri is not None)
            if row['key'] in set(PRIORITY) | extras:
                _, _, stable, _ = mf.stability(internal=True, external=False, return_status=True, tol=1e-7)
                record['independent_internal_stability'] = bool(stable)
                if not stable:
                    raise RuntimeError('Independent orbital stability check found an internal instability.')
            record['status'] = 'passed'
        except Exception as error:
            record.update(status='failed', error=str(error), traceback=traceback.format_exc())
        record['seconds'] = time.perf_counter()-tick
        report['results'].append(record)
        report.update(passed=sum(r['status'] == 'passed' for r in report['results']),
                      failed=sum(r['status'] != 'passed' for r in report['results']))
        atomic_json(args.retry/'independent-check.json', report)
        print(json.dumps(record), flush=True)
    report['status'] = 'complete' if not report['failed'] else 'complete_with_failures'
    atomic_json(args.retry/'independent-check.json', report)


if __name__ == '__main__':
    main()
