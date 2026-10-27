"""Create a venv sharing immutable dependencies, never the original editable."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def create(destination, *, references):
    subprocess.run([sys.executable,'-m','venv','--copies',str(destination)],check=True)
    output=subprocess.check_output([str(destination/'bin/python'),'-c',
        'import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip()
    import sysconfig
    original=Path(sysconfig.get_path('purelib'));target=Path(output)
    excluded=[];linked=[]
    for path in sorted(original.iterdir()):
        name=path.name
        if (name.startswith(('gradscf','__editable__')) or name.endswith(('.pth','.egg-link'))
            or name in {'__pycache__','sitecustomize.py','usercustomize.py'}
            or (not references and name.startswith(('pyscf','gpu4pyscf')))):
            excluded.append(name);continue
        if not (target/name).exists():
            (target/name).symlink_to(path,target_is_directory=path.is_dir());linked.append(name)
    (destination/'dependency-links.json').write_text(json.dumps(dict(base=sys.executable,
        original_site=str(original),venv_site=str(target),references=references,
        excluded=excluded,read_only_shared_dependencies=linked,
        note='Use PYTHONDONTWRITEBYTECODE=1. No environment copy or editable replacement.'),indent=2)+'\n')
    print(destination,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path)
    parser.add_argument('--references',action='store_true');args=parser.parse_args()
    create(args.destination,references=args.references)
