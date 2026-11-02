"""Export the exact current 500-structure pool for the NNAO paper workspace."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

from verify_training_data import EXPECTED,digest,verify


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PAPER=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper')
OUT=HERE/'payload/nnao500-20261021'


def csv_file(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    assert not OUT.exists(),'Refuse to replace an existing assembled dataset.'
    source=REPO/'artifacts/nnao-family500-20261021/inputs/references.json'
    historical=PAPER/'data/computational_examples/2026-10-21/models/update-3351/training-references.json'
    assert digest(source)==digest(historical)==EXPECTED
    lineage=json.loads((HERE/'remote-lineage.json').read_text())
    assert lineage['status']=='passed' and lineage['input_sha256']==EXPECTED
    data=json.loads(source.read_text());rows=data['structures']
    assert len(rows)==500
    OUT.mkdir(parents=True);(OUT/'structures').mkdir()
    shutil.copy2(source,OUT/'references.json')
    shutil.copy2(HERE/'remote-lineage.json',OUT/'remote-lineage.json')
    index=[];atom_rows=[];chunks=[]
    for i,row in enumerate(rows):
        label=re.sub(r'[^A-Za-z0-9_.-]+','_',row['structure_id']).strip('_')
        relative=f'structures/{i+1:04d}__{label}.xyz'
        comment=json.dumps(dict(training_index_zero_based=i,structure_id=row['structure_id'],
            charge=row['charge'],spin=row['spin'],multiplicity=row['multiplicity'],units='Angstrom'),ensure_ascii=False)
        lines=[str(row['natoms']),comment]
        lines += [symbol+' '+' '.join(format(float(x),'.17g') for x in xyz)
                  for symbol,xyz in zip(row['symbols'],row['coords_angstrom'])]
        text='\n'.join(lines)+'\n';(OUT/relative).write_text(text);chunks.append(text)
        index.append(dict(training_index_zero_based=i,structure_number_one_based=i+1,
            structure_id=row['structure_id'],formula=row['formula'],natoms=row['natoms'],
            nelectron=row['nelectron'],charge=row['charge'],spin=row['spin'],
            multiplicity=row['multiplicity'],weight=row['weight'],source_dataset=row['source_dataset'],
            source_id=row['source_id'],source_subset=row.get('subset',''),
            original_source_path=row['source_path'],role=row['role'],
            geometry_signature_sha256=row['geometry_signature_sha256'],
            xyz_file=relative,xyz_sha256=digest(OUT/relative)))
        for j,(symbol,z,xyz) in enumerate(zip(row['symbols'],row['atomic_numbers'],row['coords_angstrom'])):
            atom_rows.append(dict(training_index_zero_based=i,structure_id=row['structure_id'],
                atom_index_zero_based=j,atomic_number=z,symbol=symbol,
                x_angstrom=xyz[0],y_angstrom=xyz[1],z_angstrom=xyz[2]))
    (OUT/'all-structures.xyz').write_text(''.join(chunks))
    csv_file(OUT/'structure-index.csv',index);csv_file(OUT/'atoms.csv',atom_rows)
    source_counts=Counter(r['source_dataset'] for r in rows)
    csv_file(OUT/'source-counts.csv',[dict(source_dataset=k,structures=v) for k,v in source_counts.items()])
    coverage=[]
    for symbol in data['elements']:
        coverage.append(dict(element=symbol,structures=sum(symbol in r['symbols'] for r in rows),
            atoms=sum(r['symbols'].count(symbol) for r in rows),structure_denominator=500))
    csv_file(OUT/'element-coverage.csv',coverage)
    histogram=Counter(r['natoms'] for r in rows)
    csv_file(OUT/'atom-count-distribution.csv',[dict(natoms=n,structures=histogram[n]) for n in range(1,31)])
    summary=dict(structures=len(rows),sources=dict(source_counts),observed_elements=list(data['elements']),
        number_of_observed_elements=len({s for r in rows for s in r['symbols']}),
        total_atoms=sum(r['natoms'] for r in rows),minimum_atoms=min(histogram),maximum_atoms=max(histogram),
        isolated_atom_structures=sum(r['natoms']==1 for r in rows),
        molecule_or_complex_structures=sum(r['natoms']>1 for r in rows),
        all_charge_zero=all(r['charge']==0 for r in rows),all_spin_zero=all(r['spin']==0 for r in rows),
        all_weight_one=all(r['weight']==1 for r in rows),coordinate_units='Angstrom',
        overlap_with_old_200_by_original_metadata=sum(r['in_current_200'] for r in rows),
        source_sha256=EXPECTED,source_order_preserved=True,scf_validation_performed_by_export=False)
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'provenance.json',dict(export_date='2026-10-26',source_path=str(source),
        source_sha256=EXPECTED,historical_model_reference_path=str(historical),
        historical_reference_bit_identical=True,remote_verified=True,
        structure_order='identical to the source list; epoch shuffling is external',
        coordinate_export='17 significant decimal digits; exact float round-trip checked',
        reference_energy_labels_exported=False,basis_family_field_is_template=True,
        original_readiness_flags_preserved=True))
    shutil.copy2(HERE/'dataset_README.md',OUT/'README.md')
    shutil.copy2(HERE/'verify_training_data.py',OUT/'verify_training_data.py')
    manifest={p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write_json(OUT/'FILE_MANIFEST.json',manifest)
    report=verify(OUT);write_json(OUT/'VERIFY_REPORT.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
