import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
baseline = json.loads((root/'baseline/result.json').read_text())
rows = []
for name in ('baseline', 'optimized8', 'optimized16'):
    result = json.loads((root/name/'result.json').read_text())
    assert result['metadata']['checkpoint_sha256'] == baseline['metadata']['checkpoint_sha256']
    assert result['metadata']['structure'] == baseline['metadata']['structure']
    errors = {}
    for filename, fields in (('initial-gradient.npz', ('production_gradient', 'profile_gradient')),
                             ('training/last.npz', ('parameters', 'gradient'))):
        with np.load(root/'baseline'/filename, allow_pickle=False) as a, np.load(root/name/filename, allow_pickle=False) as b:
            for field in fields:
                np.testing.assert_allclose(a[field], b[field], atol=2e-8, rtol=2e-7)
                errors[f'{filename}:{field}'] = float(np.max(np.abs(a[field]-b[field])))
    energies = [r['loss_hartree'] for r in result['epochs']]
    np.testing.assert_allclose(energies, [r['loss_hartree'] for r in baseline['epochs']], atol=1e-8, rtol=0)
    rows.append(dict(configuration=name, full_backward_seconds=result['full_backward_median_seconds'],
        epoch_seconds=result['epoch_median_seconds'],
        backward_speedup=baseline['full_backward_median_seconds']/result['full_backward_median_seconds'],
        epoch_speedup=baseline['epoch_median_seconds']/result['epoch_median_seconds'],
        errors=errors, energies_hartree=energies))
(root/'comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
print('COMPARISON_PASSED', json.dumps(rows), flush=True)
