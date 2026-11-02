"""Select final accepted inference models, retaining provenance of excluded restarts."""
import argparse
import collections
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--quarantine',type=Path,required=True)
    args=parser.parse_args()
    root=args.models;path=root/'index.json';report=json.loads(path.read_text())
    groups=collections.defaultdict(list);excluded=[]
    for row in report['records']:
        source=Path(row['source_checkpoint'])
        if any(any(tag in part.lower() for tag in ['archive','before-restart','before-cao','original','retired','stopped-']) for part in source.parts):
            excluded.append(dict(source_checkpoint=str(source),parameter_sha256=row['parameter_sha256'],reason='historical restart or retired snapshot'))
        else:groups[str(source.parent)].append(row)
    chosen=[]
    for group,rows in groups.items():
        by_name={Path(row['source_checkpoint']).stem:row for row in rows}
        last=by_name.get('last');online=False
        if last:
            info=json.loads((root/last['metadata_path']).read_text())
            stats=info['checkpoint_statistics']
            online=any(tag in str(stats.get('training_policy','')) for tag in ['single_pass','minibatch']) or 'online' in str(stats.get('loss_kind',''))
        selected=last if online else by_name.get('best_validation',by_name.get('best',last))
        chosen.append(selected)
        for row in rows:
            if row is not selected:
                excluded.append(dict(source_checkpoint=row['source_checkpoint'],parameter_sha256=row['parameter_sha256'],reason='superseded by selected final model from same run'))
    active={row['parameter_sha256'] for row in chosen}
    args.quarantine.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,args.quarantine/'all-export-records.json')
    for folder in root.iterdir():
        if folder.is_dir() and folder.name not in active:
            target=args.quarantine/folder.name
            if target.exists():raise FileExistsError(target)
            shutil.move(str(folder),target)
    report.update(records=chosen,unique_parameter_vectors=len(active),
        selection_policy='Use latest accepted last for online/minibatch; otherwise best_validation, best, then last. Exclude retired/restart snapshots.')
    path.write_text(json.dumps(report,indent=2)+'\n')
    (root/'selection-history.json').write_text(json.dumps(dict(excluded=excluded),indent=2)+'\n')
    print('Selected',len(chosen),'final run records;',len(active),'unique vectors; excluded',len(excluded),'restart/nonfinal records.')


if __name__=='__main__':main()
