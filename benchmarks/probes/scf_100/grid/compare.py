import json
from pathlib import Path
import numpy as np
rows=[]
for device in ('cpu','gpu'):
    before=Path('baseline-'+device);after=Path('candidate-'+device)
    with np.load(before/'xc.npz') as a,np.load(after/'xc.npz') as b:
        for key in a.files:
            error=float(np.max(np.abs(a[key]-b[key])))
            passed=np.allclose(a[key],b[key],atol=1e-9,rtol=1e-10)
            rows.append(dict(device=device,key=key,max_abs=error,passed=bool(passed)))
    if device=='cpu':
        with np.load(before/'ao_samples.npz') as a,np.load(after/'ao_samples.npz') as b:
            assert np.array_equal(a['indices'],b['indices'])
            error=float(np.max(np.abs(a['values']-b['values'])))
            rows.append(dict(device=device,key='257_real_grid_samples',max_abs=error,passed=bool(np.allclose(a['values'],b['values'],atol=1e-11,rtol=1e-10))))
Path('checks.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2));assert all(r['passed'] for r in rows)
