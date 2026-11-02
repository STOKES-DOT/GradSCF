"""Render the 166 archived case-08 geometries with the installed xyzrender API.

Run with /opt/anaconda3/bin/python. No electronic-structure calculations are run.
Source XYZ files remain unchanged; only rigid display transforms are applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import xyzrender
from xyzrender.export import svg_to_png


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/computational_examples/2026-10-21"
CASE = PACKAGE / "cases/08_distance_curves_and_barriers"
TITLES = {
    "Water-Water": "Water dimer",
    "AcOH-AcOH": "Acetic acid dimer",
    "Benzene-Benzene_pi-pi": "Benzene dimer (pi-stacked)",
    "Cyclopentane-Cyclopentane": "Cyclopentane dimer",
    "Benzene-Benzene_TS": "Benzene dimer (T-shaped)",
    "Benzene-Water_OH-pi": "Benzene-water (OH-pi)",
    "1.01": "Phosphine-ammonia",
    "1.09": "Hydrogen sulfide-ammonia",
    "1.15": "Hydrogen fluoride-ammonia",
    "1.22": "Hydrogen chloride-ammonia",
    "1.28": "Hydrogen bromide-ammonia",
    "2.10": "Hydrogen sulfide-phosphoryl chloride",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory() -> list[dict]:
    data = json.loads((CASE / "inputs/dataset.json").read_text())
    index = {
        r["label"]: r
        for r in json.loads((PACKAGE / "structure-index.json").read_text())
        if r["case"] == CASE.name
    }
    records = []

    def add(structure, dataset, group_id, title, role, representative=False,
            fragments=None, scale=None):
        sid = structure["structure_id"]
        source = index[sid]
        path = PACKAGE / source["path"]
        assert path.is_file() and source["units"] == "angstrom"
        records.append(dict(
            id=sid, dataset=dataset, group_id=group_id, group_title=title,
            role=role, representative=representative, scale=scale,
            natoms=source["natoms"], charge=source["charge"],
            spin_2S=source["spin_2S"], source_xyz=str(path),
            source_xyz_sha256=sha256(path), fragments=fragments,
            geometry_sha256=source["geometry_sha256"],
        ))

    curves = defaultdict(list)
    for row in data["curves"]:
        curves[(row["dataset"], row["group"])].append(row)
    for (dataset, group_id), rows in curves.items():
        title = TITLES[group_id]
        for row in sorted(rows, key=lambda r: r["scale"]):
            add(row["structure"], dataset, group_id, title,
                f"scale = {row['scale']:.2f}", row["scale"] == 1.0,
                row["fragment_indices"], row["scale"])
        for number, monomer in enumerate(rows[0]["monomers"], 1):
            add(monomer, dataset, group_id, title, f"monomer {number}")
    for reaction in data["reactions"]:
        rid = reaction["identifier"]
        for name in [*reaction["reactants"], reaction["transition"],
                     *reaction["products"]]:
            add(data["structures"][name], "BH9", rid,
                reaction["category_name"], name[len(rid):],
                name == reaction["transition"])
    assert len(records) == len(index) == 166
    assert {r["id"] for r in records} == set(index)
    return records


def positions(molecule) -> np.ndarray:
    return np.array([v["position"] for _, v in molecule.graph.nodes(data=True)])


def render_group(records: list[dict], output: Path) -> None:
    molecules = [xyzrender.load(r["source_xyz"], charge=r["charge"],
                 multiplicity=r["spin_2S"] + 1, quick=True) for r in records]
    is_dimer_group = records[0]["dataset"] != "BH9"
    reference = next(i for i, r in enumerate(records) if r["representative"])
    if is_dimer_group:
        rotation, _ = molecules[reference].copy().orient(
            tilt_degrees=12, return_transform=True)
    projected = []
    for record, molecule in zip(records, molecules):
        if is_dimer_group:
            original = positions(molecule)
            oriented = (original - original.mean(axis=0)) @ rotation.T
            for node, point in zip(molecule.graph, oriented):
                molecule.graph.nodes[node]["position"] = tuple(point)
            molecule.oriented = True
        elif record["natoms"] > 1:
            molecule.orient(tilt_degrees=12)
        else:
            molecule.graph.nodes[next(iter(molecule.graph))]["position"] = (0., 0., 0.)
            molecule.oriented = True
        # Preserve the archived dimer partition even at compressed distances.
        removed = []
        if record["fragments"]:
            owner = {i: n for n, frag in enumerate(record["fragments"]) for i in frag}
            removed = [(a, b) for a, b in molecule.graph.edges if owner[a] != owner[b]]
            molecule.graph.remove_edges_from(removed)
        record["suppressed_interfragment_bonds"] = removed
        projected.append(positions(molecule))
    # A common pixels-per-Angstrom scale prevents auto-fit hiding separation changes.
    span = max(float(np.ptp(p[:, :2], axis=0).max()) for p in projected) + 4.0
    config = xyzrender.build_config("paton", canvas_size=900,
        atom_scale=1.65, bond_width=12, atom_stroke_width=2.5)
    config.hide_h = False
    config.auto_hide_h = False
    config.fixed_span = span
    config.fixed_center = (0., 0.)
    config.bond_orders = False
    for record, molecule in zip(records, molecules):
        stem = Path(record["source_xyz"]).stem
        relative = Path("images") / record["dataset"] / stem
        svg_path = output / (str(relative) + ".svg")
        png_path = output / (str(relative) + ".png")
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = xyzrender.render(molecule, config=deepcopy(config), orient=False,
                                    hy=True)
        svg = str(rendered)
        svg_path.write_text(svg)
        svg_to_png(svg, str(png_path), size=900, dpi=96)
        record.update(svg=str(svg_path.relative_to(output)),
                      png=str(png_path.relative_to(output)),
                      viewport_span_angstrom=span,
                      pixels_per_angstrom=(900 - 2 * config.padding) / span,
                      png_sha256=sha256(png_path), svg_sha256=sha256(svg_path))
        assert sha256(Path(record["source_xyz"])) == record["source_xyz_sha256"]
        print(f"Rendered {record['id']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    output = ROOT / (".local/case08-preview" if args.preview else "output/case08_xyzrender")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    records = inventory()
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["group_id"])].append(record)
    preview_groups = {("S66x8", "Water-Water"), ("S66x8", "Benzene-Benzene_pi-pi"),
                      ("HB300SPXx10", "1.28"), ("BH9", "BH9_08_1")}
    selected = []
    for key, group in groups.items():
        if args.preview and key not in preview_groups:
            continue
        render_group(group, output)
        selected.extend(group)
    manifest = dict(renderer=f"xyzrender {xyzrender.__version__}",
        script=str(Path(__file__).resolve()), script_sha256=sha256(Path(__file__)),
        source_dataset=str(CASE / "inputs/dataset.json"),
        source_dataset_sha256=sha256(CASE / "inputs/dataset.json"),
        units="angstrom", count=len(selected), groups=len({(r['dataset'], r['group_id']) for r in selected}),
        preview=args.preview, elapsed_seconds=time.perf_counter() - start,
        display_policy="Paton ball-and-stick; all H visible; single-line distance-perceived bonds. "
            "Dimer fragments never joined by covalent lines. No automatic TS bond assignment. "
            "Each dimer group shares a rigid camera rotation and physical display scale; "
            "BH9 species are separately oriented at a common scale within each reaction. "
            "Rigid transforms only; source XYZ bytes are unchanged.",
        records=selected)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
