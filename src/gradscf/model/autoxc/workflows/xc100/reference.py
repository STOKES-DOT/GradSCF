"""Load explicitly supplied PBE operators with their original input provenance."""
import json
from pathlib import Path

import numpy as np

from .molecule import file_hash


def reference_baseline(root, case, *, include_energy=True):
    """A fixed reference input is not relabeled as newly computed source data."""
    root = Path(root)
    receipt = json.loads((root/'reference_baselines.json').read_text())
    row = receipt['cases'][case.system_id]
    expected = {'archive_sha256':case.archive_hash,'metadata_sha256':case.metadata_hash,
                'basis_sha256':case.metadata['canonical_basis_sha256']}
    if (receipt['floor']!=case.floor or
            receipt['pbe_model_sha256']!=file_hash(Path(__file__).with_name('pbe.py')) or
            any(row[name]!=value for name,value in expected.items())):
        raise ValueError('Reference PBE model or target identity changed')
    path = root/'reference/cache/cases'/case.system_id/'pbe.npz'
    if file_hash(path)!=row['cache_sha256']:
        raise ValueError('Reference PBE operator checksum changed')
    with np.load(path,allow_pickle=False) as stored:
        if str(stored['core'])!=receipt['origin_core'] or str(stored['fingerprint'])!=row['fingerprint']:
            raise ValueError('Reference PBE operator provenance changed')
        energy, potential = (float(stored['energy']) if include_energy else None),stored['potential']
    if (potential.shape!=case.target.density.shape or potential.dtype!=np.float64 or
            (energy is not None and not np.isfinite(energy)) or not np.isfinite(potential).all()):
        raise ValueError('Reference PBE operators require finite float64 arrays in the target AO space')
    return energy,potential
