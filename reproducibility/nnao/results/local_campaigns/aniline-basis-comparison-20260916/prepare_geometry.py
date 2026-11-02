"""Generate a fixed aniline benchmark geometry; no quantum optimization."""
import json
from pathlib import Path
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors
root=Path(__file__).resolve().parent
mol=Chem.AddHs(Chem.MolFromSmiles('Nc1ccccc1'))
settings=AllChem.ETKDGv3();settings.randomSeed=20260916
assert AllChem.EmbedMolecule(mol,settings)==0
status=AllChem.MMFFOptimizeMolecule(mol,mmffVariant='MMFF94',maxIters=2000)
assert status==0,status
symbols=[a.GetSymbol() for a in mol.GetAtoms()]
coords=mol.GetConformer().GetPositions().tolist()
record=dict(name='aniline',formula=rdMolDescriptors.CalcMolFormula(mol),symbols=symbols,
            coords_angstrom=coords,charge=0,spin=0,
            geometry_source='RDKit ETKDGv3 seed 20260916, MMFF94 optimization, maxIters 2000',
            geometry_note='Fixed force-field geometry, not a quantum-optimized geometry.',
            rdkit_version=rdkit.__version__,mmff_status=status)
assert record['formula']=='C6H7N'
(root/'geometry.json').write_text(json.dumps(record,indent=2)+'\n')
(root/'aniline.xyz').write_text(str(len(symbols))+'\n'+record['geometry_source']+'\n'+''.join(f'{s:2s} {x:.12f} {y:.12f} {z:.12f}\n' for s,(x,y,z) in zip(symbols,coords)))
print(record['formula'],len(symbols),'atoms; MMFF status',status)
