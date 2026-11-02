"""Data-only handoff of the three ethane radial-response figures to NNAO_paper."""
import hashlib
import json
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
PAPER = Path("/Users/jiaoyuan/Documents/GitHub/NNAO_paper")
RELATIVE = Path("2026-10-27/cases/16_ethane_spd_radial_response")
DESTINATION = PAPER / "data/computational_examples" / RELATIVE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((SOURCE / "verification.json").read_text())["status"] == "passed"
    assert json.loads((SOURCE / "portability-check.json").read_text())["status"] == "passed"
    files = []
    for folder in ("inputs", "figures", "tables", "arrays"):
        files.extend(p for p in (SOURCE / folder).rglob("*") if p.is_file())
    files.extend(SOURCE / p for p in ("README.md", "analyze.py", "input-manifest.json",
                 "verification.json", "verification-supplement.json", "portability-check.json"))
    # An existing, different artifact is never silently overwritten.
    for path in files:
        target = DESTINATION / path.relative_to(SOURCE)
        if target.exists():
            assert digest(path) == digest(target), str(target)
    manifest = []
    for path in sorted(files):
        relative = path.relative_to(SOURCE)
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        assert digest(path) == digest(target)
        manifest.append(dict(file=str(relative), bytes=target.stat().st_size, sha256=digest(target)))
    (DESTINATION / "FILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    index = PAPER / "data/computational_examples/README.md"
    original = index.read_text()
    row = ("| Ethane C0 s/p/d radial response and all contraction coefficients, Case 16 | "
           f"[2026-10-27 analysis]({RELATIVE}/README.md) | "
           "25 torsion angles + corrected 56 stretches; update 3351 MACE vs free contraction; "
           "three channel figures; no new SCF |\n")
    anchor = "|---|---|---|\n"
    if str(RELATIVE) not in original:
        assert original.count(anchor) == 1
        (SOURCE / "paper-index-before.md").write_text(original)
        updated = original.replace(anchor, anchor + row, 1)
        assert updated.replace(row, "", 1) == original
        index.write_text(updated)
    assert (DESTINATION / "README.md").exists()
    report = dict(status="passed", destination=str(DESTINATION), files=len(manifest),
                  bytes=sum(r["bytes"] for r in manifest),
                  index_change="one new Case 16 row; other content preserved",
                  existing_case_files_modified=0, manuscript_files_modified=0,
                  existing_output_figures_modified=0)
    (SOURCE / "INSTALL_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
