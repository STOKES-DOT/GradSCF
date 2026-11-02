"""Verify released file hashes, training structures, and inference-only payloads."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='Hash every data file; default verifies training inputs and weights.')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'data-manifest.json').read_text())
    if args.full:
        for row in manifest['files']:
            path=ROOT/row['path']
            if not path.is_file() or path.stat().st_size!=row['bytes']:
                raise RuntimeError('Missing/truncated data (run git lfs pull): '+str(path))
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=row['sha256']:raise RuntimeError('Data checksum mismatch: '+str(path))
            if row.get('compression')=='gzip':
                with gzip.open(path,'rb') as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=row['source_sha256']:
                        raise RuntimeError('Decompressed source mismatch: '+str(path))
        print('Verified',len(manifest['files']),'file hashes.')
    path=ROOT/'paper/data/training/nnao500-20261021/references.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='5e1b21c837f318da1bedc0b55f3f2bbfefc45218390928ce1b217db0607dc299':
        raise RuntimeError('The canonical ordered 500-structure input changed.')
    rows=json.loads(path.read_text())['structures']
    assert len(rows)==len({r['structure_id'] for r in rows})==500
    assert sum(len(r['symbols']) for r in rows)==5402
    for row in rows:
        coordinates=np.asarray(row['coords_angstrom'])
        assert coordinates.shape==(len(row['symbols']),3) and np.isfinite(coordinates).all()
        assert row['charge']==row['spin']==0
        assert row['nelectron']==sum(row['atomic_numbers']) and row['nelectron']%2==0
    seen=set();count=0
    for group in ['local_checkpoints','remote_checkpoints']:
        folder=ROOT/'models'/group;index=json.loads((folder/'index.json').read_text())
        assert not index['errors']
        for item in index['records']:
            path=folder/item['parameters_path']
            if path in seen:continue
            seen.add(path)
            with np.load(path,allow_pickle=False) as saved:
                arrays={key:saved[key] for key in saved.files}
            assert set(arrays) in ({'parameters'},{'theta','beta'},{'theta'})
            assert all(np.isfinite(a).all() for a in arrays.values())
            if set(arrays)=={'parameters'}:
                digest=hashlib.sha256(arrays['parameters'].tobytes(order='C')).hexdigest()
            else:
                h=hashlib.sha256()
                for key,array in sorted(arrays.items()):
                    h.update(key.encode());h.update(str(array.dtype).encode())
                    h.update(str(array.shape).encode());h.update(array.tobytes(order='C'))
                digest=h.hexdigest()
            assert digest==item['parameter_sha256'],path
            count+=1
    canonical=json.loads((ROOT/'models/canonical-nine.json').read_text())
    assert set(canonical)=={f'nnao_{z}p_{s}' for z in ['sz','dz','tz'] for s in ['s','m','l']}
    benchmark=json.loads((ROOT/'models/benchmark-nine.json').read_text())
    assert set(benchmark)==set(canonical)
    assert all(row['matched_inference_record'] is not None for row in benchmark.values())
    print('Verified canonical 500 structures, all nine model families, and',count,'inference-only parameter payloads.')


if __name__=='__main__':main()
