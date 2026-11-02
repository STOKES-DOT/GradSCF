"""Locate every affected archived point, retaining only rechecked endpoints."""
import json
from pathlib import Path
import numpy as np
from reoptimize import prepare, evaluator, normalize, OUT, SOURCE
from gradscf.model.nnao.runtime import atomic_json


def main():
    target = OUT / 'archived-native-audit.json'
    report = json.loads(target.read_text()) if target.exists() else dict(status='running', results=[])
    done = {r['index'] for r in report['results']}
    original = json.loads((SOURCE / 'scan-summary.json').read_text())['points']
    for index in range(56):
        if index in done:
            continue
        row, layout, old, experiment, manifest = prepare(index)
        evaluate, _ = evaluator(experiment, layout)
        mace_energy, _, mace_info = evaluate(old['mace_coefficients'], gradient=False)
        c = normalize(old['coefficients'], layout)
        energy, gradient, info = evaluate(c)
        norm = float(np.linalg.norm(gradient[np.asarray(old['active_mask'])]))
        result = dict(index=index, distance_angstrom=row['distance_angstrom'],
            mace_energy_hartree=mace_energy, mace_scf=mace_info,
            free_energy_hartree=energy, free_scf=info, gradient_norm=norm,
            archived_free_energy_hartree=original[index]['free_energy_hartree'],
            free_energy_lowering_hartree=original[index]['free_energy_hartree']-energy,
            mace_energy_lowering_hartree=original[index]['mace_energy_hartree']-mace_energy,
            reusable=bool(norm <= 1e-5 and abs(original[index]['free_energy_hartree']-energy) <= 1e-8))
        report['results'].append(result)
        report['status'] = 'complete' if index == 55 else 'running'
        atomic_json(target, report)
        print(index, row['distance_angstrom'], result['reusable'], norm,
              result['free_energy_lowering_hartree'], flush=True)


if __name__ == '__main__':
    main()
