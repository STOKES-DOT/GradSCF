"""Verify the data handoff without importing GradSCF or rerunning calculations."""
import csv
import hashlib
import json
from pathlib import Path
import sys


def verify(root: Path) -> dict:
    manifest = json.loads((root / "FILE_MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = root / row["path"]
        assert path.stat().st_size == row["bytes"], row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    case = root / "cases/05_ethane_cc_stretch"
    bench = case / "artifacts/nnao-ethane-dissociation-benchmark-20261022"
    repair = case / "artifacts/nnao-ethane-dissociation-repair-20261022"
    curves = list(csv.DictReader((bench / "curves.csv").open()))
    structures = json.loads((bench / "structures.json").read_text())
    native = json.loads((bench / "nnao-results.json").read_text())["points"]
    checks = json.loads((repair / "independent-validation.json").read_text())
    assert len(curves) == 504 and all(r["accepted"] == "True" for r in curves)
    names = sorted({r["basis"] for r in curves})
    assert len(names) == 9
    assert all(sum(r["basis"] == name for r in curves) == 56 for name in names)
    assert len(structures) == len(native) == 56
    assert len({s["distance_angstrom"] for s in structures}) == 56
    assert checks["status"] == "complete" and checks["passed"] == 112 and checks["failed"] == 0
    assert all(r["status"] == "passed" for r in checks["results"])
    assert max(abs(r["error_hartree"]) for r in checks["results"]) < 1e-8
    for i, (structure, point) in enumerate(zip(structures, native)):
        assert structure["distance_angstrom"] == point["distance_angstrom"]
        assert structure["charge"] == structure["spin"] == 0
        assert point["status"] == "passed" and point["free_gradient_converged"]
        assert point["gradient_norm"] < 1e-5
        xyz = (case / f"structures/distance-{i:03d}.xyz").read_text().splitlines()
        assert int(xyz[0]) == len(structure["symbols"]) == 8
        for line, symbol, coords in zip(xyz[2:], structure["symbols"], structure["coords_angstrom"]):
            parts = line.split()
            assert parts[0] == symbol and list(map(float, parts[1:])) == coords
        final = repair / f"final-bases/distance-{i:03d}"
        basis_rows = [json.loads(line) for line in (final / "frozen-bases.jsonl").read_text().splitlines()]
        assert len(basis_rows) == 2
        for row in basis_rows:
            assert row["symbols"] == structure["symbols"]
            assert row["coords_angstrom"] == structure["coords_angstrom"]
            assert row["nao"] == 44
            field = "mace_energy_hartree" if row["basis"] == "MACE-update3351" else "free_energy_hartree"
            assert abs(row["native_energy_hartree"] - point[field]) < 1e-10
        for label, field in (("NNAO / MACE", "mace_energy_hartree"), ("NNAO / free contraction", "free_energy_hartree")):
            row = next(r for r in curves if r["basis"] == label and int(r["index"]) == i)
            assert float(row["energy_hartree"]) == point[field]
    return {
        "status": "passed",
        "verified_payload_files": len(manifest["files"]),
        "curve_rows": len(curves),
        "curve_count": len(names),
        "structures_xyz": len(structures),
        "frozen_basis_records": 112,
        "independent_checks": checks["passed"],
        "independent_max_abs_error_hartree": checks["max_abs_error_hartree"],
        "failed_reference_candidate_seeds": checks["failed_seed_count"],
        "coordinate_roundtrip": "exact float equality",
        "recalculation": False,
    }


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root), indent=2))
