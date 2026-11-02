"""Build a content manifest and a searchable model/dataset catalog for the release."""
import csv
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    records={}
    for name in ['local-data-manifest.json','paper-figures-manifest.json','remote-data-manifest.json']:
        source=json.loads((ROOT/name).read_text())
        for row in source['files']:
            if row['path'] in records and records[row['path']]['sha256']!=row['sha256']:
                raise ValueError('Conflicting data identities: '+row['path'])
            records[row['path']]={**row,'collection_manifest':name}
    models=[];canonical={}
    for collection in ['local_checkpoints','remote_checkpoints']:
        directory=ROOT/'models'/collection
        index=json.loads((directory/'index.json').read_text())
        if index['errors']:raise ValueError(index['errors'])
        for item in index['records']:
            info=json.loads((directory/item['metadata_path']).read_text())
            identity=info.get('identity',{});config=identity.get('model_config',{})
            checkpoint=info['source_checkpoint'];parts=Path(checkpoint).parts
            family=config.get('basis_family',identity.get('basis_family',''))
            row=dict(collection=collection,family=family,channels=config.get('channels',''),
                hidden_irreps=config.get('hidden_irreps',''),
                parameter_count=info['parameter_count'],parameter_sha256=item['parameter_sha256'],
                parameters_path=str((directory/item['parameters_path']).relative_to(ROOT)),
                metadata_path=str((directory/item['metadata_path']).relative_to(ROOT)),
                source_checkpoint=checkpoint,selection=Path(checkpoint).stem,
                epoch=info['checkpoint_statistics'].get('epoch',''),
                update_step=info['checkpoint_statistics'].get('update_step',''))
            models.append(row)
            if collection=='remote_checkpoints' and checkpoint.endswith('/last.npz'):
                if '/GradSCF_family500_20261021/' in checkpoint and '/runs/' in checkpoint and Path(checkpoint).parent.name=='train':
                    label=parts[parts.index('runs')+1]
                    if label.startswith('nnao_'):canonical[label]=row
                elif checkpoint.endswith('/GradSCF_mace500_20261014/artifacts/mace500-20261014/train/last.npz'):
                    canonical['nnao_szp_l']=row
        for path in directory.rglob('*'):
            if path.is_file():
                relative=str(path.relative_to(ROOT))
                records[relative]=dict(path=relative,sha256=digest(path),bytes=path.stat().st_size,
                    collection_manifest='inference_export',source='see model metadata/index.json')
    (ROOT/'models/canonical-nine.json').write_text(json.dumps(canonical,indent=2)+'\n')
    with (ROOT/'models/catalog.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(models[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(models)
    datasets=[]
    for relative in sorted(records):
        path=ROOT/relative
        if path.name!='references.json':continue
        definition=json.loads(path.read_text());rows=definition.get('structures') if isinstance(definition,dict) else None
        if not isinstance(rows,list):continue
        identifiers=[row.get('structure_id') for row in rows]
        datasets.append(dict(path=relative,count=len(rows),unique_ids=len(set(identifiers)),
            units=definition.get('units'),method=definition.get('method'),basis_family=definition.get('basis_family'),
            elements=sorted({s for row in rows for s in row.get('symbols',[])}),
            max_atoms=max((len(row.get('symbols',[])) for row in rows),default=0)))
    (ROOT/'dataset-catalog.json').write_text(json.dumps(datasets,indent=2)+'\n')
    unique={row['sha256']:row['bytes'] for row in records.values()}
    report=dict(format='nnao-paper-data-manifest-v1',files=list(records.values()),
        file_count=len(records),bytes=sum(row['bytes'] for row in records.values()),
        unique_payloads=len(unique),unique_bytes=sum(unique.values()),
        inference_records=len(models),unique_parameter_vectors=len({row['parameter_sha256'] for row in models}),
        canonical_families=sorted(canonical),dataset_definitions=len(datasets))
    (ROOT/'data-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__':main()
