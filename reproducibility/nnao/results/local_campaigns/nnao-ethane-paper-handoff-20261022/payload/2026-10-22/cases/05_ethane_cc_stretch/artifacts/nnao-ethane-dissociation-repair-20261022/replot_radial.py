"""Rebuild radial figures from the accepted, branch-corrected frozen bases."""
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT.parent / 'nnao-ethane-dissociation-benchmark-20261022'
SOURCE = ROOT.parent / 'nnao-ethane-dissociation-20261019'
FINAL = ROOT / 'final-bases'
os.environ.setdefault('MPLCONFIGDIR', str(BENCHMARK / '.mpl-cache'))


def main():
    points = json.loads((BENCHMARK / 'nnao-results.json').read_text())['points']
    shutil.copy2(SOURCE / 'structures.json', FINAL / 'structures.json')
    for point in points:
        result = dict(point, gap_kcal_mol=(point['mace_energy_hartree']-point['free_energy_hartree'])*627.5094740631)
        (FINAL / f'distance-{point["index"]:03d}/result.json').write_text(json.dumps(result, indent=2)+'\n')
    spec = importlib.util.spec_from_file_location('repaired_radial_analysis', SOURCE / 'analyze_radial.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    analysis.OUT = FINAL
    analysis.main()
    for name in ('carbon-radial-functions', 'carbon-radial-response'):
        for ext in ('png', 'pdf'):
            shutil.copy2(FINAL / f'{name}.{ext}', BENCHMARK / f'{name}-repaired.{ext}')
    for name in ('radial-metrics', 'energy-comparison'):
        shutil.copy2(FINAL / f'{name}.csv', BENCHMARK / f'{name}-repaired.csv')


if __name__ == '__main__':
    main()
