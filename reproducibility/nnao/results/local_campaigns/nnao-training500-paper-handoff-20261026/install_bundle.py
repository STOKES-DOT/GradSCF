"""Install the structure-only dataset and narrow data-index additions."""
import json
from pathlib import Path
import shutil

from verify_training_data import digest,verify


HERE=Path(__file__).resolve().parent
PAPER=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper')
SOURCE=HERE/'payload/nnao500-20261021'
DESTINATION=PAPER/'data/training/nnao500-20261021'
INDEX=PAPER/'data/computational_examples/README.md'


def fingerprint(root):
    return {p.relative_to(root).as_posix():digest(p) for p in root.rglob('*')
            if p.is_file() and p.name!='.DS_Store'}


def main():
    assert not DESTINATION.exists(),'Refuse to replace an existing training dataset.'
    assert not (DESTINATION.parent/'README.md').exists(),'Review an existing training index.'
    assert digest(INDEX)==digest(HERE/'index-before.md'),'Data index changed; refresh the insertion.'
    source_check=verify(SOURCE)
    protected={name:fingerprint(PAPER/name) for name in (
        'manuscript','output','scripts','data/computational_examples/2026-10-21',
        'data/computational_examples/2026-10-22','data/computational_examples/2026-10-26')}
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(SOURCE,DESTINATION)
    destination_check=verify(DESTINATION)
    (DESTINATION.parent/'README.md').write_bytes((HERE/'training-index.md').read_bytes())
    INDEX.write_bytes((HERE/'index-after.md').read_bytes())
    unchanged={name:fingerprint(PAPER/name)==value for name,value in protected.items()}
    assert all(unchanged.values()),'Protected content changed during installation; inspect concurrent work.'
    assert fingerprint(SOURCE)==fingerprint(DESTINATION)
    report=dict(status='passed',destination=str(DESTINATION),source_check=source_check,
        destination_check=destination_check,copied_files=sum(p.is_file() for p in DESTINATION.rglob('*')),
        copied_bytes=sum(p.stat().st_size for p in DESTINATION.rglob('*') if p.is_file()),
        protected_content_verified_unchanged=unchanged,protected_files_verified=sum(map(len,protected.values())),
        edited_manuscript=False,changed_training_state=False,
        written_indices=[str(INDEX),str(DESTINATION.parent/'README.md')])
    (HERE/'INSTALL_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
