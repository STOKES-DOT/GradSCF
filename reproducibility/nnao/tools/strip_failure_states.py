"""Keep failure diagnostics while quarantining nonaccepted parameter snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--quarantine',type=Path,required=True)
    args=parser.parse_args()
    manifest_path=args.root/args.manifest
    report=json.loads(manifest_path.read_text())
    records=[];count=0
    for row in report['files']:
        relative=Path(row['path'])
        if relative.suffix!='.npz' or 'failed' not in relative.name.lower():
            records.append(row);continue
        source=args.root/relative
        quarantine=args.quarantine/relative
        incoming=source if source.exists() else quarantine
        with np.load(incoming,allow_pickle=False) as saved:
            text=str(saved['metadata']) if 'metadata' in saved else json.dumps({'payload_keys':saved.files})
        json.loads(text)
        target=source.with_suffix('.metadata.json')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(text+'\n')
        raw=target.read_bytes()
        records.append(dict(path=str(target.relative_to(args.root)),source=row['source'],
            source_sha256=row['source_sha256'],source_bytes=row['source_bytes'],
            sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
            transformation='diagnostic metadata only; nonaccepted parameter vector quarantined'))
        report['excluded_training_states'].append(dict(source=row['source'],
            reason='nonaccepted_failure_parameter_snapshot',bytes=row['bytes'],sha256=row['sha256']))
        if source.exists():
            quarantine.parent.mkdir(parents=True,exist_ok=True)
            if quarantine.exists():raise FileExistsError(quarantine)
            shutil.move(str(source),str(quarantine))
        count+=1
    report.update(files=records,file_count=len(records),bytes=sum(r['bytes'] for r in records))
    manifest_path.write_text(json.dumps(report,indent=2)+'\n')
    print('Preserved metadata for',count,'nonaccepted snapshots; raw vectors quarantined.')


if __name__=='__main__':main()
