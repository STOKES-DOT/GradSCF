"""Install only the new data supplement and data index; do not edit prose."""
import hashlib
import json
from pathlib import Path
import shutil

from verify_export import verify


HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "payload/2026-10-22"
DATA = Path("/Users/jiaoyuan/Documents/GitHub/NNAO_paper/data/computational_examples")
DESTINATION = DATA / "2026-10-22"


def fingerprint(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file() and p.name != ".DS_Store"
    }


def main():
    assert not DESTINATION.exists(), "Refuse to replace an existing dated supplement."
    assert not (DATA / "README.md").exists(), "Review an existing data index before modifying it."
    source_check = verify(PAYLOAD)
    old = DATA / "2026-10-21"
    before = fingerprint(old)
    shutil.copytree(PAYLOAD, DESTINATION)
    destination_check = verify(DESTINATION)
    assert fingerprint(old) == before, "Prior archive changed during installation."
    with (DATA / "README.md").open("x") as stream:
        stream.write((HERE / "computational_examples_README.md").read_text())
    report = {
        "status": "passed",
        "destination": str(DESTINATION),
        "data_index": str(DATA / "README.md"),
        "source_check": source_check,
        "destination_check": destination_check,
        "prior_archive_files_verified_unchanged": len(before),
        "copied_files": sum(p.is_file() for p in DESTINATION.rglob("*")),
        "copied_bytes": sum(p.stat().st_size for p in DESTINATION.rglob("*") if p.is_file()),
        "edited_manuscript": False,
        "edited_output_figures": False,
    }
    (HERE / "INSTALL_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
