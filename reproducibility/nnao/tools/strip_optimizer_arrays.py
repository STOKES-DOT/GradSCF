"""Remove legacy optimizer arrays while preserving all scientific outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


def strip(root,manifest_name,quarantine):
    root=Path(root);quarantine=Path(quarantine)
    path=root/manifest_name;report=json.loads(path.read_text());changes=[]
    for row in report['files']:
        source=root/row['path']
        if source.suffix!='.npz':continue
        with zipfile.ZipFile(source) as archive:keys={n.removesuffix('.npy') for n in archive.namelist()}
        remove={key for key in keys if any(s in key.lower() for s in ('optimizer','adam','opt_state'))}
        if not remove:continue
        with np.load(source,allow_pickle=False) as saved:
            arrays={key:saved[key] for key in saved.files if key not in remove}
        temporary=source.with_name(source.stem+'.inference-only.npz')
        np.savez(temporary,**arrays)
        with np.load(temporary,allow_pickle=False) as check:
            assert set(check.files)==set(arrays)
            for key,array in arrays.items():
                np.testing.assert_array_equal(check[key],array)
                assert check[key].dtype==array.dtype and check[key].tobytes()==array.tobytes()
        target=quarantine/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.move(str(source),target);temporary.replace(source)
        before=dict(row)
        with source.open('rb') as stream:row['sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
        row['bytes']=source.stat().st_size
        row['transformation']='Removed optimizer arrays '+','.join(sorted(remove))+'; all remaining arrays verified identical.'
        changes.append(dict(path=row['path'],removed_keys=sorted(remove),original_sha256=before['sha256'],sha256=row['sha256']))
    report['bytes']=sum(r['bytes'] for r in report['files'])
    report.setdefault('optimizer_stripping',[]).extend(changes)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('Stripped legacy optimizer arrays from',len(changes),'scientific records; all other arrays preserved.')
    return changes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--quarantine',type=Path,required=True)
    args=parser.parse_args();strip(args.root,args.manifest,args.quarantine)


if __name__=='__main__':main()
