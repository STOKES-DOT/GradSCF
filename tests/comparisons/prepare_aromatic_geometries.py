"""Seeded fixed force-field geometries for the three aromatic regressions."""
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,smiles in [('toluene','Cc1ccccc1'),('nitrobenzene','O=[N+]([O-])c1ccccc1')]:
        mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
        cfg=AllChem.ETKDGv3();cfg.randomSeed=20260920
        assert AllChem.EmbedMolecule(mol,cfg)==0
        assert AllChem.MMFFOptimizeMolecule(mol,maxIters=2000)==0
        data=dict(name=name,symbols=[a.GetSymbol() for a in mol.GetAtoms()],coords_angstrom=mol.GetConformer().GetPositions().tolist(),
                  charge=0,spin=0,geometry_source='ETKDGv3 seed 20260920 + MMFF94, fixed for every method')
        (output/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    old=Path('artifacts/aniline-basis-comparison-20260916/geometry.json')
    (output/'aniline.json').write_bytes(old.read_bytes())


if __name__=='__main__':
    import sys
    prepare(sys.argv[1])
