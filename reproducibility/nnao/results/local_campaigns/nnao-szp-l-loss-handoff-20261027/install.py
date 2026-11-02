"""Install a verified, dedicated loss-data entry; preserve the old Case 11."""
import hashlib
import json
import re
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
PAYLOAD = OUT / "payload"
PAPER = Path("/Users/jiaoyuan/Documents/GitHub/NNAO_paper")
DESTINATION = PAPER / "data/training/nnao-szp-l-500-20261014"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((PAYLOAD / "verification.json").read_text())["status"] == "passed"
    files = [f for f in PAYLOAD.rglob("*") if f.is_file()]
    assert not any("cache" in f.parts for f in files)
    for source in files:
        target = DESTINATION / source.relative_to(PAYLOAD)
        if target.exists():
            assert sha(source) == sha(target), str(target)
    manifest = []
    for source in sorted(files):
        relative = source.relative_to(PAYLOAD)
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        assert sha(source) == sha(target)
        manifest.append(dict(file=str(relative), sha256=sha(target), bytes=target.stat().st_size))
    (DESTINATION / "FILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for link in re.findall(r"\]\(([^)]+)\)", (DESTINATION / "README.md").read_text()):
        assert (DESTINATION / link).is_file(), link
    index = PAPER / "data/computational_examples/README.md"
    original = index.read_text()
    link = "../training/nnao-szp-l-500-20261014/README.md"
    row = ("| NNAO-SZP-L 500-system training loss | "
           f"[Dedicated loss data and plots]({link}) | "
           "Epochs 0–596; 592 complete online means; epoch 4 missing; "
           "partial epoch 597 excluded; source job 99005; Case 11 preserved |\n")
    anchor = "|---|---|---|\n"
    if link not in original:
        assert original.count(anchor) == 1
        (OUT / "paper-index-before.md").write_text(original)
        updated = original.replace(anchor, anchor + row, 1)
        assert updated.replace(row, "", 1) == original
        assert index.read_text() == original
        index.write_text(updated)
    report = dict(status="passed", destination=str(DESTINATION), files=len(manifest),
        bytes=sum(r["bytes"] for r in manifest), index_change="one loss-data row only",
        old_case11_modified=False, dataset_manifest_modified=False, manuscript_modified=False,
        training_state_modified=False)
    (OUT / "INSTALL_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
