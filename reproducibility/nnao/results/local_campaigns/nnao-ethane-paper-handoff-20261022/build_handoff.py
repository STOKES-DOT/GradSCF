"""Prepare an inspectable data-only supplement before copying to NNAO_paper."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from verify_export import verify


SOURCE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROOT = HERE / "payload/2026-10-22"
CASE = ROOT / "cases/05_ethane_cc_stretch"
PAPER = Path("/Users/jiaoyuan/Documents/GitHub/NNAO_paper")
BENCH_NAME = "nnao-ethane-dissociation-benchmark-20261022"
REPAIR_NAME = "nnao-ethane-dissociation-repair-20261022"
IMPORTED = {}
EXCLUDED = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def copy_sources(name):
    source = SOURCE / "artifacts" / name
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        skip = any(part.startswith("cache-") or part in {"__pycache__", ".mpl-cache", "review", "archive-hcore-only"} for part in relative.parts)
        skip |= path.name.startswith("._") or path.name == ".DS_Store"
        skip |= name == BENCH_NAME and relative.as_posix() in {"run.py", "run.log", "plot.log"}
        if skip:
            EXCLUDED.append({"source": str(path), "bytes": path.stat().st_size, "reason": "Integral/build/render cache or obsolete hcore-only duplicate; historical source retained in GradSCF/prior archive."})
            continue
        target = CASE / "artifacts" / name / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        IMPORTED[target.relative_to(ROOT).as_posix()] = str(path)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert not ROOT.exists(), "Use a new staging location; never replace an existing handoff implicitly."
    for name in (BENCH_NAME, REPAIR_NAME):
        copy_sources(name)
    bench = CASE / "artifacts" / BENCH_NAME
    repair = CASE / "artifacts" / REPAIR_NAME
    structures = json.loads((bench / "structures.json").read_text())
    points = json.loads((bench / "nnao-results.json").read_text())["points"]
    validation = json.loads((repair / "independent-validation.json").read_text())
    index = []
    for i, structure in enumerate(structures):
        target = CASE / f"structures/distance-{i:03d}.xyz"
        target.parent.mkdir(parents=True, exist_ok=True)
        comment = json.dumps({"index": i, "structure_id": structure["structure_id"], "distance_angstrom": structure["distance_angstrom"], "charge": structure["charge"], "spin_2S": structure["spin"], "units": "angstrom", "geometry_optimized": False}, ensure_ascii=False)
        rows = [str(len(structure["symbols"])), comment]
        rows.extend(symbol + " " + " ".join(format(float(x), ".17g") for x in coords) for symbol, coords in zip(structure["symbols"], structure["coords_angstrom"]))
        target.write_text("\n".join(rows) + "\n")
        final = repair / f"final-bases/distance-{i:03d}/frozen-bases.jsonl"
        records = [json.loads(line) for line in final.read_text().splitlines()]
        index.append({"index": i, "structure_id": structure["structure_id"], "distance_angstrom": structure["distance_angstrom"], "natoms": 8, "charge": structure["charge"], "spin_2S": structure["spin"], "xyz": target.relative_to(CASE).as_posix(), "source": str(SOURCE / "artifacts" / BENCH_NAME / "structures.json"), "source_record": i, "frozen_basis": final.relative_to(CASE).as_posix(), "raw_frozen_record_structure_ids": [r["structure_id"] for r in records]})
    write_json(CASE / "structure-index.json", index)
    write_csv(CASE / "structure-index.csv", [{k: v for k, v in row.items() if k != "raw_frozen_record_structure_ids"} for row in index])
    compare = [{k: row[k] for k in ("index", "distance_angstrom", "basis", "native_energy_hartree", "independent_energy_hartree", "error_hartree", "selected_start", "status")} for row in validation["results"]]
    write_csv(CASE / "tables/pyscf-comparison.csv", compare)
    branches = [{"index": row["index"], "distance_angstrom": row["distance_angstrom"], "basis": row["basis"], **start} for row in validation["results"] for start in row["starts"]]
    fields = list(dict.fromkeys(key for row in branches for key in row))
    write_csv(CASE / "tables/pyscf-initial-guesses.csv", branches, fields)
    critical = [{k: p[k] for k in ("index", "distance_angstrom", "mace_energy_hartree", "free_energy_hartree", "gradient_norm", "free_gradient_converged", "status")} for p in points if 2.7 <= p["distance_angstrom"] <= 2.9]
    write_csv(CASE / "tables/critical-points.csv", critical)
    model = PAPER / "data/computational_examples/2026-10-21/models/update-3351/parameters.npz"
    model_ref = {"path_from_date_root": "../2026-10-21/models/update-3351/parameters.npz", "source": str(model), "sha256": digest(model), "bytes": model.stat().st_size, "model_update": 3351, "copied_again": False}
    catalog = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE),
        "status": "complete_data_only_handoff",
        "cases": [{"id": "05_ethane_cc_stretch", "title": "Corrected restricted-HF ethane C-C stretching: nine basis/mode curves", "archive_path": "cases/05_ethane_cc_stretch", "status": "complete_corrected", "supersedes": "../2026-10-21/cases/05_ethane_cc_stretch", "cohort": "MACE-update-3351", "method": "All-electron DF-RHF / def2-universal-jkfit; spherical AO; float64", "distances": 56, "curve_count": 9, "accepted_curve_points": 504, "reoptimized_free_points": 32, "rechecked_reused_free_points": 24, "independent_frozen_basis_checks": 112, "independent_max_abs_error_hartree": validation["max_abs_error_hartree"], "failed_reference_candidate_seeds": validation["failed_seed_count"], "coefficient_gradient_tolerance": 1e-5, "free_optimization_gradient": "existing implicit SCF", "finite_differences": False, "limitations": ["Rigid restricted closed-shell RHF stress test; not physical homolytic dissociation or a bond dissociation energy.", "Finite-budget local contraction optimization; no proof of global minima.", "PySCF independently recomputed fixed-basis energies, not independent contraction optimization or gradient checks.", "Some raw frozen-basis structure_id fields retain historical ethane-torsion labels. Join by point index, distance, and exact coordinates using structure-index.json."]}],
        "model_reference": model_ref,
        "units": {"distance": "angstrom", "absolute_energy": "hartree", "relative_energy": "kcal/mol", "radial_grid": "bohr"},
        "relative_energy": "E(R)-E(R0), each curve with its own reference at R0=1.5259763767561005 angstrom",
        "prior_archive_modified": False,
        "paper_output_figures_regenerated": False,
        "manuscript_modified": False,
        "calculations_rerun_for_handoff": False,
    }
    write_json(ROOT / "catalog.json", catalog)
    write_json(ROOT / "EXCLUDED_FILES.json", {"files": EXCLUDED, "total_bytes": sum(row["bytes"] for row in EXCLUDED), "full_four_index_eri_added": False})
    (ROOT / "_tools").mkdir(parents=True)
    shutil.copy2(HERE / "verify_export.py", ROOT / "_tools/verify_export.py")
    (CASE / "README.md").write_text(CASE_README)
    (ROOT / "README.md").write_text(ROOT_README)
    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file():
            relative = path.relative_to(ROOT).as_posix()
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path), "source": IMPORTED.get(relative), "kind": "imported" if relative in IMPORTED else "generated"})
    write_json(ROOT / "FILE_MANIFEST.json", {"algorithm": "sha256", "files": files})
    checksummed = [p for p in sorted(ROOT.rglob("*")) if p.is_file()]
    (ROOT / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in checksummed))
    report = verify(ROOT)
    write_json(ROOT / "VERIFY_REPORT.json", report)
    with (ROOT / "SHA256SUMS").open("a") as stream:
        stream.write(f"{digest(ROOT / 'VERIFY_REPORT.json')}  VERIFY_REPORT.json\n")
    (HERE / "computational_examples_README.md").write_text(INDEX_README)
    print(json.dumps({**report, "total_files": sum(p.is_file() for p in ROOT.rglob("*")), "total_bytes": sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()), "staging": str(ROOT)}, indent=2))


ROOT_README = """# NNAO computational examples — corrected ethane supplement, 2026-10-22

Data-only handoff for the manuscript Codex. This package **supersedes Case 05
(ethane C-C stretch) in the 2026-10-21 archive**. Other cases remain unchanged.
The old archive, manuscript and paper-specific output figures were not edited.

- [Corrected case and reading order](cases/05_ethane_cc_stretch/README.md)
- [Machine-readable catalog and limitations](catalog.json)
- [Export checks](VERIFY_REPORT.json)
- [Original-source hashes](FILE_MANIFEST.json), [payload checksums](SHA256SUMS)
- [Omitted numerical/build caches](EXCLUDED_FILES.json)

The corrected set has 56 distances and nine curves: 504 accepted energy points,
112 independently checked frozen NNAO bases, 56 exact-coordinate XYZ files,
final coefficients/bases, optimization histories and corrected radial plots.
No SCF, optimization, training or finite differences were rerun for this handoff.
The existing inference-only update-3351 model is referenced in `catalog.json`
under `../2026-10-21/models/update-3351/`; it is not duplicated.

Verify this supplement without a chemistry/JAX environment:

```bash
python _tools/verify_export.py
```

Use this corrected Case 05 for new paper figures. The older
`output/ethane_cc_stretch/` still represents the historical uncorrected data;
it was deliberately left for the manuscript/figure thread to regenerate.
"""

CASE_README = """# Corrected ethane C-C stretching: MACE, free contraction and seven conventional bases

## Status and replacement

**Complete: 504/504 accepted energy points.** This supplement replaces the
2026-10-21 Case 05 interpretation, including the five unconverged endpoints
at 2.70–2.90 angstrom and the hcore-induced energy jump. Historical inputs and
failed/candidate records are retained; no interpolation or smoothing was used.

## Methods and numerical evidence

The same 56 rigid, neutral singlet C2H6 geometries span 1.525976–8 angstrom.
All calculations use all-electron DF-RHF / def2-universal-jkfit, spherical AOs
and float64. MACE update 3351, its predicted raw coefficients, primitive
exponents and fixed core are unchanged. NNAO has 44 AOs.

The hcore guess can converge to a higher internally stable RHF solution.
At 2.75 angstrom in the same frozen MACE basis, PySCF hcore gives
-78.73767289499133 Eh, while MINAO gives -79.00203112890705 Eh; corrected
GradSCF gives -79.00203112890696 Eh. Internal stability alone does not identify
the global lowest RHF branch.

All NNAO geometries were rechecked with hcore/neutral-population guesses.
Free contraction was reoptimized at 32 distances (2.70–8 angstrom), using two
coefficient starts; 24 shorter-distance endpoints were rechecked and reused.
The accepted Euclidean-normalized coefficient-gradient criterion remains
1e-5. The current implicit SCF gradient is used. Not every rejected/local
optimizer candidate converged; endpoint acceptance and all histories are explicit.

Seven conventional bases contribute 392 accepted geometry–basis cases:
STO-3G, STO-6G, custom STO-6G+C(d)/H(p), 3-21G, 6-31G(d,p), cc-pVDZ,
and cc-pVTZ. Multiple SCF guesses were considered; 392 is not an SCF-call count.
The custom polarization exponents are C(d)=0.8 and H(p)=1.1 bohr^-2.

Independent CPU PySCF 2.9.0 recomputation passed all 112 frozen NNAO basis
cases: maximum absolute difference 1.137e-12 Eh (acceptance tolerance 1e-8 Eh).
One MINAO candidate failed numerically; other guesses resolved the same case.
This validates fixed-basis energies, not independent contraction optimization
or agreement of PySCF and GradSCF gradients. Native runs used local macOS
arm64 / JAX 0.8.1 CPU. No full four-index ERI cache is included.

## Data and figures

- [All 504 values](artifacts/nnao-ethane-dissociation-benchmark-20261022/curves.csv)
  and [AO/energy summary](artifacts/nnao-ethane-dissociation-benchmark-20261022/summary.csv).
- [Relative curves](artifacts/nnao-ethane-dissociation-benchmark-20261022/ethane-cc-relative-repaired.pdf),
  [absolute curves](artifacts/nnao-ethane-dissociation-benchmark-20261022/ethane-cc-absolute-repaired.pdf),
  [critical interval and gradients](artifacts/nnao-ethane-dissociation-benchmark-20261022/ethane-cc-critical-repaired.pdf).
  PNG previews are beside each PDF.
- [Corrected carbon radial functions](artifacts/nnao-ethane-dissociation-benchmark-20261022/carbon-radial-functions-repaired.pdf),
  [radial response](artifacts/nnao-ethane-dissociation-benchmark-20261022/carbon-radial-response-repaired.pdf),
  and [224 radial-function metrics](artifacts/nnao-ethane-dissociation-benchmark-20261022/radial-metrics-repaired.csv).
- `structures/`: 56 standard XYZ files with exact coordinate float roundtrips,
  Angstrom units, charge=0 and 2S=0. [Structure index](structure-index.json)
  maps each point to its source, distance and final basis.
- `artifacts/nnao-ethane-dissociation-repair-20261022/final-bases/`: per-point
  `frozen-bases.jsonl`, `best-coefficients.npz`, geometry and endpoint checks.
  Raw frozen-basis `structure_id` fields can retain historical torsion labels;
  use the point index, distance and exact coordinates in the structure index.
- [Flat PySCF comparison](tables/pyscf-comparison.csv),
  [all PySCF initial guesses, including failures](tables/pyscf-initial-guesses.csv),
  [five corrected endpoints](tables/critical-points.csv), and
  [full independent checks](artifacts/nnao-ethane-dissociation-repair-20261022/independent-validation.json).
- Full repair drivers, optimizer histories, 56 archived-endpoint audits and
  source hashes are retained under `artifacts/`; primitive integral, compiled
  and rendering caches are omitted. Raw absolute paths are source provenance,
  not portable replay commands. Numerical replay requires the recorded GradSCF environment.

## Units and interpretation

Absolute energies are Eh; relative energies are kcal/mol using
627.5094740631 kcal mol^-1 / Eh. Every relative curve uses its own E(R0), with
R0=1.5259763767561005 angstrom. Radial coordinates are bohr.

This is a restricted closed-shell RHF rigid-stretch stress test, **not physical
homolytic dissociation or a bond dissociation energy**. Finite-budget free
contraction is not a certified global minimum, and cc-pVTZ is a same-method
basis reference, not an exact solution. Paper-specific figures were not regenerated.
"""

INDEX_README = """# NNAO computational examples: current data index

Data-only handoffs for the manuscript/figure thread. Source records, units,
numerical status and limitations are documented in each dated package.

| Scope | Current package | Relationship to older data |
|---|---|---|
| Ethane C-C rigid stretch, Case 05 | [2026-10-22 corrected Case 05](2026-10-22/cases/05_ethane_cc_stretch/README.md) | Supersedes 2026-10-21 Case 05: corrected 56-point MACE/free curves, seven conventional bases, radial functions and 112 PySCF checks |
| Other computational examples, Cases 01–04 and 06–13 | [2026-10-21 archive](2026-10-21/README.md) | Unchanged |
| Frozen update-3351 inference model | `2026-10-21/models/update-3351/` | Shared by both handoffs; not retrained or duplicated |

For the corrected ethane case, use [504 curve values](2026-10-22/cases/05_ethane_cc_stretch/artifacts/nnao-ethane-dissociation-benchmark-20261022/curves.csv),
[56 geometries and final-basis mapping](2026-10-22/cases/05_ethane_cc_stretch/structure-index.json),
and the [flat PySCF comparison](2026-10-22/cases/05_ethane_cc_stretch/tables/pyscf-comparison.csv).
The previous five-point gap and large hcore-branch jump are superseded.

The historical `2026-10-21/` archive and its checksums remain unchanged.
`output/ethane_cc_stretch/` still uses the older data until the figure thread
explicitly regenerates it. This handoff does not edit the manuscript, figures
in `output/`, or training state.
"""


if __name__ == "__main__":
    main()
