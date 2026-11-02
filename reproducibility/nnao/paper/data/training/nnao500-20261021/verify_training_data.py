"""Verify training structures, order, state labels and lossless XYZ exports."""
import csv
import hashlib
import json
from pathlib import Path


EXPECTED='5e1b21c837f318da1bedc0b55f3f2bbfefc45218390928ce1b217db0607dc299'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root=Path(root)
    assert digest(root/'references.json')==EXPECTED
    definition=json.loads((root/'references.json').read_text())
    rows=definition['structures']
    assert len(rows)==500 and len({r['structure_id'] for r in rows})==500
    manifest=json.loads((root/'FILE_MANIFEST.json').read_text())
    for name,expected in manifest.items():assert digest(root/name)==expected,name
    with (root/'structure-index.csv').open() as stream:index=list(csv.DictReader(stream))
    assert len(index)==500
    chunks=[]
    for ordinal,(row,record) in enumerate(zip(rows,index)):
        assert int(record['training_index_zero_based'])==ordinal
        assert record['structure_id']==row['structure_id']
        assert int(record['charge'])==row['charge']==0
        assert int(record['spin'])==row['spin']==0
        assert row['units']=='Angstrom' and 1<=row['natoms']<=30
        assert len(row['symbols'])==len(row['coords_angstrom'])==row['natoms']
        assert sum(row['atomic_numbers'])-row['charge']==row['nelectron']
        assert row['nelectron']%2==0
        text=(root/record['xyz_file']).read_text();lines=text.splitlines()
        assert len(lines)==row['natoms']+2 and int(lines[0])==row['natoms']
        for line,symbol,xyz in zip(lines[2:],row['symbols'],row['coords_angstrom']):
            fields=line.split();assert fields[0]==symbol
            assert list(map(float,fields[1:]))==xyz,(row['structure_id'],line)
        chunks.append(text)
    assert (root/'all-structures.xyz').read_text()==''.join(chunks)
    with (root/'atoms.csv').open() as stream:atoms=list(csv.DictReader(stream))
    expected=[(i,r['structure_id'],j,z,s,*xyz) for i,r in enumerate(rows)
              for j,(z,s,xyz) in enumerate(zip(r['atomic_numbers'],r['symbols'],r['coords_angstrom']))]
    actual=[(int(r['training_index_zero_based']),r['structure_id'],int(r['atom_index_zero_based']),
             int(r['atomic_number']),r['symbol'],float(r['x_angstrom']),float(r['y_angstrom']),
             float(r['z_angstrom'])) for r in atoms]
    assert actual==expected
    lineage=json.loads((root/'remote-lineage.json').read_text())
    assert lineage['status']=='passed' and lineage['input_sha256']==EXPECTED
    assert all(r['ordered_structures_equal'] for r in lineage['metadata_checks'])
    return dict(status='passed',structures=500,unique_structure_ids=500,xyz_files=500,
        atomic_coordinate_rows=len(atoms),natoms_min=min(r['natoms'] for r in rows),
        natoms_max=max(r['natoms'] for r in rows),
        observed_elements=len({s for r in rows for s in r['symbols']}),
        exact_coordinate_roundtrip=True,original_order_preserved=True,
        source_sha256=EXPECTED,files_verified=len(manifest),new_scf_calculations=False)


if __name__=='__main__':
    import sys
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent),indent=2))
