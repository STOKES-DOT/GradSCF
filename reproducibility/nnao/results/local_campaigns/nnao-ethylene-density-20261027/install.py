"""Install only the verified ethylene comparison data into NNAO_paper."""
import hashlib
import json
import re
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
PAPER = Path("/Users/jiaoyuan/Documents/GitHub/NNAO_paper")
RELATIVE = Path("2026-10-27/cases/17_ethylene_density_planes")
DESTINATION = PAPER / "data/computational_examples" / RELATIVE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    check = json.loads((SOURCE / "verification.json").read_text())
    assert check["status"] == "passed" and check["density_maps"] == 12
    assert json.loads((SOURCE / "portability-check.json").read_text())["status"] == "passed"
    plot = json.loads((SOURCE / "plot-metadata.json").read_text())
    assert len(plot["files"]) == 28 and all((SOURCE / f).is_file() for f in plot["files"])
    files = []
    for folder in ("inputs", "optimization", "states", "arrays", "figures"):
        files.extend(f for f in (SOURCE / folder).rglob("*") if f.is_file())
    for name in ("README.md", "run.py", "density.py", "plot.py", "verify.py", "configuration.json",
                 "environment.json", "results.json", "summary.csv", "model-lineage-check.json",
                 "pool-summary.json", "verification.json", "plot-metadata.json", "portability-check.json", "run.log"):
        files.append(SOURCE / name)
    for path in files:
        target = DESTINATION / path.relative_to(SOURCE)
        if target.exists():
            assert sha(path) == sha(target), f"Existing different file: {target}"
    manifest = []
    for path in sorted(files):
        relative = path.relative_to(SOURCE)
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        assert sha(path) == sha(target)
        manifest.append(dict(file=str(relative), bytes=target.stat().st_size, sha256=sha(target)))
    (DESTINATION / "FILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for link in re.findall(r"\]\(([^)]+)\)", (DESTINATION / "README.md").read_text()):
        assert (DESTINATION / link).exists(), link
    index = PAPER / "data/computational_examples/README.md"
    original = index.read_text()
    row = ("| Ethylene total-density planes, Case 17 | "
           f"[2026-10-27 six-basis comparison]({RELATIVE}/README.md) | "
           "xy molecular and xz perpendicular planes; 12 maps; update 3351 MACE/free + four conventional bases; "
           "matched DF-RHF and shared color scale |\n")
    anchor = "|---|---|---|\n"
    if str(RELATIVE) not in original:
        assert original.count(anchor) == 1
        (SOURCE / "paper-index-before.md").write_text(original)
        updated = original.replace(anchor, anchor + row, 1)
        assert updated.replace(row, "", 1) == original
        index.write_text(updated)
    report = dict(status="passed", destination=str(DESTINATION), files=len(manifest),
                  bytes=sum(f["bytes"] for f in manifest),
                  index_change="one new Case 17 row only", manuscript_files_modified=0,
                  existing_case_files_modified=0, training_state_modified=False)
    (SOURCE / "INSTALL_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
