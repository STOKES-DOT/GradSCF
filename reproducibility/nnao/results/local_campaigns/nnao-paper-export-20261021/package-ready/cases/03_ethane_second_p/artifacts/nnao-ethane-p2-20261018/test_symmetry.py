"""Exact atom mapping between reflected rigid ethane torsion geometries."""
from pathlib import Path
import json
import numpy as np
from symmetry import mirror_atom_map


def test_mirror_mapping_preserves_elements_and_all_pair_distances():
    rows=json.loads((Path(__file__).resolve().parent.parent/'nnao-ethane-torsion-20261018'/'structures.json').read_text())
    lookup={int(r['angle_deg']):r for r in rows}
    for angle in (0,5,25,55):
        source,target=lookup[angle],lookup[120-angle]
        permutation,error=mirror_atom_map(source,target)
        assert sorted(permutation.tolist())==list(range(8))
        assert [source['symbols'][i] for i in permutation]==target['symbols']
        assert error<1e-11
        x=np.asarray(source['coords_angstrom'])[permutation]
        y=np.asarray(target['coords_angstrom'])
        np.testing.assert_allclose(np.linalg.norm(x[:,None]-x[None],axis=2),
            np.linalg.norm(y[:,None]-y[None],axis=2),atol=1e-11,rtol=0)
