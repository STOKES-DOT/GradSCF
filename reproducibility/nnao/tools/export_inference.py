"""Export accepted parameter vectors and model identities, excluding Adam/gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def export(candidates, output, *, merge=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records, errors = [], []
    if merge and (output/'index.json').exists():
        previous=json.loads((output/'index.json').read_text())
        retry={row['path'] for row in candidates}
        records=[row for row in previous['records'] if row['source_checkpoint'] not in retry]
        errors=[row for row in previous['errors'] if row['source_checkpoint'] not in retry]
    for index,row in enumerate(candidates):
        path=Path(row['path'])
        before=path.stat()
        try:
            with np.load(path,allow_pickle=False) as saved:
                primary='parameters' if 'parameters' in saved else 'theta'
                if primary not in saved:
                    raise ValueError('No parameter vector in this checkpoint format.')
                arrays={primary:np.asarray(saved[primary])}
                if 'beta' in saved:
                    arrays['beta']=np.asarray(saved['beta'])
                parameters=arrays[primary]
                if any(not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError('Non-finite or non-numeric parameters.')
                metadata=json.loads(str(saved['metadata'])) if 'metadata' in saved else {}
            parameter_hash=hashlib.sha256(parameters.tobytes(order='C')).hexdigest()
            if set(arrays)!={'parameters'}:
                digest=hashlib.sha256()
                for key,array in sorted(arrays.items()):
                    digest.update(key.encode());digest.update(str(array.dtype).encode())
                    digest.update(str(array.shape).encode());digest.update(array.tobytes(order='C'))
                parameter_hash=digest.hexdigest()
            folder=output/parameter_hash
            folder.mkdir(exist_ok=True)
            destination=folder/'parameters.npz'
            if not destination.exists():
                temporary=folder/'parameters.tmp.npz'
                np.savez(temporary,**arrays)
                temporary.replace(destination)
            identity=dict(metadata.get('identity',{}))
            caches=identity.pop('caches',None)
            if caches is not None:
                identity['cache_identities']={key:{k:value[k] for k in
                    ('signature','cache_sha256','pool_nao','contracted_nao','df_shape') if k in value}
                    for key,value in caches.items()}
            info=dict(format='nnao-inference-parameters-v1',identity=identity,
                source_checkpoint=str(path),source_checkpoint_bytes=before.st_size,
                source_metadata_sha256=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest(),
                parameter_sha256=parameter_hash,parameter_shape=list(parameters.shape),
                parameter_dtype=str(parameters.dtype),parameter_count=sum(int(a.size) for a in arrays.values()),
                payload_keys=sorted(arrays),
                optimizer_included=False,gradient_included=False,
                checkpoint_statistics={k:metadata[k] for k in
                    ('epoch','update_step','accepted_epoch','loss_hartree','loss_kind',
                     'parameters_evaluated','training_policy') if k in metadata})
            info_hash=hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest()
            record_path=folder/(info_hash+'.json')
            record_path.write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError('Checkpoint changed during export.')
            with np.load(destination,allow_pickle=False) as check:
                assert set(check.files)==set(arrays)
                for key,array in arrays.items():
                    np.testing.assert_array_equal(check[key],array)
            records.append(dict(source_checkpoint=str(path),parameters_path=str(destination.relative_to(output)),
                metadata_path=str(record_path.relative_to(output)),parameter_sha256=parameter_hash,
                parameter_count=sum(int(a.size) for a in arrays.values()),bytes=destination.stat().st_size))
        except Exception as error:
            errors.append(dict(source_checkpoint=str(path),error=str(error)))
        if (index+1)%10==0:
            print(index+1,'/',len(candidates),'exported',len(records),'errors',len(errors),flush=True)
    report=dict(format='nnao-inference-index-v1',records=records,errors=errors,
        unique_parameter_vectors=len({r['parameter_sha256'] for r in records}))
    (output/'index.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Finished',len(records),'records;',report['unique_parameter_vectors'],'unique vectors;',len(errors),'errors',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--merge',action='store_true',help='Replace only the listed source records in an existing index.')
    args=parser.parse_args()
    export(json.loads(args.candidates.read_text()),args.output,merge=args.merge)


if __name__=='__main__':main()
