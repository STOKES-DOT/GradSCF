"""Install the verified data-only supplement; preserve existing paper work."""
import json
from pathlib import Path
import shutil

from verify_bundle import digest,verify


HERE=Path(__file__).resolve().parent
PAPER=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper')
DATA=PAPER/'data/computational_examples'
PAYLOAD=HERE/'payload/2026-10-26'
DESTINATION=DATA/'2026-10-26'
CASE='cases/14_bsse_counterpoise'


def fingerprint(root):
    if root.is_file():return {root.name:digest(root)}
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*')
            if p.is_file() and p.name!='.DS_Store'}


def main():
    assert not DESTINATION.exists(),'Refuse to replace an existing dated supplement.'
    before_index=HERE/'data-index-before.md';after_index=HERE/'data-index-after.md'
    assert digest(DATA/'README.md')==digest(before_index),'Data index changed; regenerate the narrow insertion.'
    source_check=verify(PAYLOAD/CASE)
    protected={name:fingerprint(PAPER/name) for name in (
        'manuscript','output','scripts','data/computational_examples/2026-10-21',
        'data/computational_examples/2026-10-22')}
    shutil.copytree(PAYLOAD,DESTINATION)
    destination_check=verify(DESTINATION/CASE)
    (DATA/'README.md').write_bytes(after_index.read_bytes())
    unchanged={name:fingerprint(PAPER/name)==before for name,before in protected.items()}
    assert all(unchanged.values()),'Concurrent changes detected in protected content; inspect the report inputs.'
    # Verify every copied file, including the dated landing README and report
    # that are intentionally outside the self-referential case manifest.
    assert fingerprint(PAYLOAD)==fingerprint(DESTINATION)
    report=dict(status='passed',destination=str(DESTINATION),case=str(DESTINATION/CASE),
        data_index=str(DATA/'README.md'),source_check=source_check,destination_check=destination_check,
        copied_files=sum(p.is_file() for p in DESTINATION.rglob('*')),
        copied_bytes=sum(p.stat().st_size for p in DESTINATION.rglob('*') if p.is_file()),
        protected_content_verified_unchanged=unchanged,
        protected_files_verified=sum(len(v) for v in protected.values()),
        edited_manuscript=False,edited_existing_output=False,changed_training_state=False)
    (HERE/'INSTALL_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
