"""Freeze ten GMTKN55 v2 structures disjoint from previous NNAO selections."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


SOURCE_SHA256 = "ee90a5ed52a58d1d1b78ae8b602b63dc2db8f6f98174ae7ef22a09a8a4e57d87"
PILOT_SHA256 = "cddae74070ee5053656c1e105be45e0b8dbd98ab15ec347d0fc87e4233405899"
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
EXPECTED_IDS = (
    "W4-11/n2o", "W4-11/hooh", "W4-11/n2h4", "W4-11/f2co",
    "PA26/c2f6", "DARC/furane", "HAL59/pyr", "S22/03", "S66/05",
    "HAL59/12_benF6-benA",
)
DESCRIPTIONS = {
    "W4-11/n2o": ("nitrous oxide", "Small polar N/O multiple-bond environment."),
    "W4-11/hooh": ("hydrogen peroxide", "O-O bond and oxygen lone pairs."),
    "W4-11/n2h4": ("hydrazine", "N-N single bond and amine lone pairs."),
    "W4-11/f2co": ("carbonyl fluoride", "Carbonyl with two strongly polar C-F bonds."),
    "PA26/c2f6": ("hexafluoroethane", "Fluorine-rich saturated carbon environment."),
    "DARC/furane": ("furan", "Oxygen-containing aromatic heterocycle."),
    "HAL59/pyr": ("pyridine", "Nitrogen-containing aromatic heterocycle."),
    "S22/03": ("formic acid dimer", "Hydrogen-bonded carboxylic acid complex."),
    "S66/05": ("methanol dimer", "Hydrogen-bonded alcohol complex."),
    "HAL59/12_benF6-benA": ("hexafluorobenzene", "Fluorinated aromatic ring."),
}
SELECTION_NOTE = (
    "Structure IDs, exact coordinate signatures and whole molecular formulas exclude "
    "stage-two train200/validation50, pilot24 and prior aniline training (C6H7N). "
    "This does not ensure fragment, chemical graph or reaction-connected isolation; "
    "for example methanol monomer occurs in the old pilot. No reference reaction labels "
    "are used. These structures have no absolute reference energy in the source dataset."
)


def formula_key(row):
    return tuple(sorted(Counter(row["symbols"]).items()))


def select_rows(source_rows, split_rows, *, expected_ids=EXPECTED_IDS):
    """Reject any requested structure with ID, geometry or formula leakage."""
    source = {row["structure_id"]: row for row in source_rows}
    if len(source) != len(source_rows):
        raise ValueError("Duplicate structure IDs in the source dataset.")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Duplicate requested structure IDs.")
    missing = set(expected_ids) - source.keys()
    if missing:
        raise ValueError(f"Missing requested structure IDs: {sorted(missing)}")
    excluded = [row for rows in split_rows.values() for row in rows]
    used_ids = {row["structure_id"] for row in excluded}
    used_coordinates = {row["coordinate_signature_sha256"] for row in excluded}
    used_formulas = {formula_key(row) for row in excluded}
    used_formulas.add((("C", 6), ("H", 7), ("N", 1)))  # Prior aniline training.
    selected = []
    seen_formulas = set()
    for identifier in expected_ids:
        row = source[identifier]
        signature = row.get("coordinate_signature_sha256")
        key = formula_key(row)
        if identifier in used_ids:
            raise ValueError(f"Structure ID overlap: {identifier}")
        if not signature or signature in used_coordinates:
            raise ValueError(f"Missing or overlapping coordinate signature: {identifier}")
        if key in used_formulas or key in seen_formulas:
            raise ValueError(f"Whole formula overlap (including prior aniline): {identifier}")
        symbols = row["symbols"]
        if (row.get("charge") != 0 or row.get("spin") != 0
                or not symbols or not set(symbols) <= ATOMIC_NUMBERS.keys()
                or row.get("natoms") != len(symbols)):
            raise ValueError(f"Invalid neutral closed-shell CHNOF structure: {identifier}")
        nelectron = sum(ATOMIC_NUMBERS[symbol] for symbol in symbols)
        if nelectron % 2 or row.get("nelectron") != nelectron:
            raise ValueError(f"Invalid even electron count: {identifier}")
        coordinates = row.get("coords_angstrom", [])
        if len(coordinates) != len(symbols) or any(
                len(point) != 3 or not all(math.isfinite(x) for x in point)
                for point in coordinates):
            raise ValueError(f"Invalid coordinates: {identifier}")
        counts = dict(key)
        order = (["C", "H"] if "C" in counts else []) + [
            symbol for symbol in sorted(counts) if "C" not in counts or symbol not in ("C", "H")]
        formula = "".join(symbol + (str(counts[symbol]) if counts[symbol] != 1 else "")
                          for symbol in order if symbol in counts)
        chemical_name, reason = DESCRIPTIONS.get(identifier, (identifier, "Independent test structure."))
        selected.append(dict(row, name=identifier, split="test", formula=formula,
            formula_group="".join(f"{symbol}{count}" for symbol, count in key),
            chemical_name=chemical_name, selection_reason=reason,
            expected_nnao_nao=4 * symbols.count("H") + 10 * (len(symbols) - symbols.count("H")),
            selection_note=SELECTION_NOTE))
        seen_formulas.add(key)
    return selected


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/Volumes/TF/GradSCF_GMTKN55"))
    parser.add_argument("--stage2-selection", type=Path,
                        default=Path("artifacts/nnao-stage2-20261007/selection"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    source = args.dataset_root / "data/v2/structures.jsonl"
    pilot = args.dataset_root / "selections/hf_pilot_20261007/structures.jsonl"
    paths = {"train": args.stage2_selection / "train.jsonl",
             "validation": args.stage2_selection / "validation.jsonl", "pilot": pilot}
    if sha256(source) != SOURCE_SHA256 or sha256(pilot) != PILOT_SHA256:
        raise ValueError("Source or pilot snapshot differs from the verified selection.")
    split_rows = {name: read_rows(path) for name, path in paths.items()}
    if {name: len(rows) for name, rows in split_rows.items()} != {"train": 200, "validation": 50, "pilot": 24}:
        raise ValueError("Expected the original train200, validation50 and pilot24 selections.")
    parent_manifest = args.stage2_selection / "selection.json"
    if json.loads(parent_manifest.read_text())["source_structures_sha256"] != SOURCE_SHA256:
        raise ValueError("Stage-two manifest source differs from the benchmark source.")
    selected = select_rows(read_rows(source), split_rows)
    payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in selected)
    manifest = dict(source_version="v2", source_path=str(source.resolve()),
        source_sha256=SOURCE_SHA256, source_commit="ccabc16c8c4677fc60c19870ae96ac25405844a2",
        selection_script_sha256=sha256(__file__), selected_structures_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        exclusions={name: dict(path=str(path.resolve()), sha256=sha256(path),
                              count=len(split_rows[name])) for name, path in paths.items()},
        parent_selection_manifest_sha256=sha256(parent_manifest),
        prior_single_molecule_formula_exclusions=["C6H7N"], structure_ids=list(EXPECTED_IDS),
        count=len(selected), isolation_checks=["structure_id", "coordinate_signature_sha256", "whole_formula"],
        coordinates_unit="angstrom", charge=0, spin=0, method="RHF", basis="szp663_direct",
        core_primitives=6, cartesian=False, selection_note=SELECTION_NOTE,
        no_calculation_or_training_performed=True,
        systems=[{key: row[key] for key in ("structure_id", "chemical_name", "formula", "natoms", "nelectron", "expected_nnao_nao")}
                 for row in selected])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"test-selection.jsonl": payload,
               "selection-manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n"}
    if any((args.output_dir / name).exists() for name in outputs):
        raise FileExistsError("Refusing to overwrite an existing benchmark selection.")
    for name, content in outputs.items():
        with (args.output_dir / name).open("x") as stream:
            stream.write(content)
    print(json.dumps(dict(count=len(selected), selected_structures_sha256=manifest["selected_structures_sha256"],
                         output_dir=str(args.output_dir.resolve()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
