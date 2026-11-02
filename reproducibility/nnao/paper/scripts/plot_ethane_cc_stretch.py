"""Plot archived restricted-HF ethane C-C stretching energies and status flags.

Only figure construction is performed; no SCF or contraction optimization runs.
"""
from __future__ import annotations

import csv
from io import BytesIO
import json

import numpy as np
from PIL import Image, ImageChops
import xyzrender
from xyzrender.export import svg_to_png_bytes
from plot_case08_reactions import (
    ROOT, PPT, OUTPUT_ROOT, LINE_SCALE, MARKER_SCALE,
    configure_style, sha, style_metadata,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PACKAGE = ROOT / "data/computational_examples/2026-10-22"
CASE = PACKAGE / "cases/05_ethane_cc_stretch"
ARCHIVE = CASE / "artifacts/nnao-ethane-dissociation-benchmark-20261022"
SOURCE = ARCHIVE / "curves.csv"
SUMMARY = ARCHIVE / "nnao-results.json"
TRADITIONAL = ARCHIVE / "traditional-results.json"
STRUCTURES = ARCHIVE / "structures.json"
OUT = OUTPUT_ROOT / "ethane_cc_stretch"
HARTREE_KCAL_MOL = 627.5094740631
SERIES = [
    ("STO-3G", "STO-3G", "#A7ADB4", "--", None, 1.3),
    ("STO-6G", "STO-6G", "#9A8268", "-.", None, 1.3),
    ("STO-6G + C(d), H(p)", "STO-6G + pol.", "#8A56A2", ":", None, 1.7),
    ("3-21G", "3-21G", "#676D75", "-.", None, 1.4),
    ("6-31G(d,p)", "6-31G(d,p)", "#8D933E", "-.", None, 1.7),
    ("cc-pVDZ", "cc-pVDZ", "#5C8C9E", ":", None, 1.7),
    ("cc-pVTZ", "cc-pVTZ", "#0057B8", "-", "o", 2.8),
    ("NNAO / MACE", "NNAO/MACE", "#ef233c", "--", "s", 1.8),
    ("NNAO / free contraction", "Free (44 AO)", "#171717", "-", "o", 1.7),
]


def stretch_structure():
    structure_path = STRUCTURES
    structure = json.loads(structure_path.read_text())[0]
    index = json.loads((CASE / "structure-index.json").read_text())
    entry = next(r for r in index if r["index"] == 0)
    assert entry["distance_angstrom"] == structure["distance_angstrom"]
    xyz_path = CASE / entry["xyz"]
    original_hash = sha(xyz_path)
    xyz = np.array(structure["coords_angstrom"])
    tx, tz = np.deg2rad([42., -90.])
    rx = np.array([[1., 0., 0.], [0., np.cos(tx), -np.sin(tx)], [0., np.sin(tx), np.cos(tx)]])
    rz = np.array([[np.cos(tz), -np.sin(tz), 0.], [np.sin(tz), np.cos(tz), 0.], [0., 0., 1.]])
    camera = rz @ rx
    display = xyz @ camera.T
    cc_display = display[1] - display[0]
    assert cc_display[0] > 0 and abs(cc_display[1]) < 1e-12
    molecule = xyzrender.load(xyz_path, charge=0, multiplicity=1, quick=True)
    for node, point in zip(molecule.graph, display):
        molecule.graph.nodes[node]["position"] = tuple(point)
    molecule.oriented = True
    config = xyzrender.build_config("paton", canvas_size=900, atom_scale=1.85,
                                   bond_width=12, atom_stroke_width=2.5)
    config.hide_h = config.auto_hide_h = False
    config.fixed_span, config.fixed_center = 6.4, (0., 0.)
    config.bond_orders = False
    svg = str(xyzrender.render(molecule, config=config, orient=False, hy=True))
    scale = (900 - 2 * config.padding) / config.fixed_span
    px = 450 + scale * display[:, 0]
    py = 450 - scale * display[:, 1]
    arrow_y = max(py) + 80
    midpoint = (px[0] + px[1]) / 2
    left_head, right_head = px[0] - 95, px[1] + 95
    arrow_width = 8 if PPT else 5
    font_weight = ' font-weight="bold"' if PPT else ""
    arrows = f'''<defs><marker id="stretch-outward-head" markerWidth="20" markerHeight="16"
        refX="17" refY="8" orient="auto" markerUnits="userSpaceOnUse">
        <path d="M0,0 L20,8 L0,16 Z" fill="#4b5563"/></marker></defs>
        <path d="M{midpoint-30:.3f},{arrow_y:.3f} L{left_head:.3f},{arrow_y:.3f}"
        fill="none" stroke="#4b5563" stroke-width="{arrow_width}" marker-end="url(#stretch-outward-head)"/>
        <path d="M{midpoint+30:.3f},{arrow_y:.3f} L{right_head:.3f},{arrow_y:.3f}"
        fill="none" stroke="#4b5563" stroke-width="{arrow_width}" marker-end="url(#stretch-outward-head)"/>
        <text x="{midpoint:.3f}" y="{arrow_y+14:.3f}" font-family="DejaVu Sans"
        font-style="italic" font-size="42"{font_weight} text-anchor="middle" fill="#171717">R</text>'''
    svg = svg.replace("</svg>", arrows + "</svg>")
    (OUT / "ethane-stretch-structure.svg").write_text(svg)
    png = svg_to_png_bytes(svg, size=2700)
    image = Image.open(BytesIO(png)).convert("RGB")
    image.save(OUT / "ethane-stretch-structure.png")
    box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    image = image.crop((box[0] - 30, box[1] - 30, box[2] + 30, box[3] + 30))
    assert sha(xyz_path) == original_hash
    return image, dict(source_xyz=str(xyz_path), source_xyz_sha256=original_hash,
        source_structure=str(structure_path), source_structure_sha256=sha(structure_path), source_structure_index=0,
        reference_distance_angstrom=structure["distance_angstrom"],
        camera_rotation=camera.tolist(), cc_display_vector=cc_display.tolist(),
        arrows="Two outward arrows; symmetric rigid methyl-group separation")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    rows = list(csv.DictReader(SOURCE.open()))
    for r in rows:
        for k in ("distance_angstrom", "energy_hartree", "plot_energy_hartree", "relative_energy_kcal_mol"):
            r[k] = float(r[k])
        r["index"], r["nao"] = int(r["index"]), int(r["nao"])
        assert r["accepted"] == "True" and r["status"] == "passed"
        assert r["energy_hartree"] == r["plot_energy_hartree"]
    summary = json.loads(SUMMARY.read_text())
    by_key = {(r["basis"], r["index"]): r for r in rows}
    assert len(rows) == len(by_key) == 504 and len(summary["points"]) == 56
    for point in summary["points"]:
        assert point["status"] == "passed" and point["free_gradient_converged"]
        assert point["gradient_norm"] <= 1e-5
        for basis, energy_key, scf_key in [("NNAO / MACE", "mace_energy_hartree", "mace_scf"),
                                            ("NNAO / free contraction", "free_energy_hartree", "free_scf")]:
            row = by_key[(basis, point["index"])]
            assert row["distance_angstrom"] == point["distance_angstrom"]
            assert row["energy_hartree"] == point[energy_key]
            assert point[scf_key]["converged"] and point[scf_key]["stable"]
    traditional = json.loads(TRADITIONAL.read_text())["results"]
    assert len(traditional) == 392
    for point in traditional:
        assert point["status"] == "passed"
        assert by_key[(point["basis"], point["index"])]["energy_hartree"] == point["energy_hartree"]
    distance = np.array([by_key[("NNAO / MACE", i)]["distance_angstrom"] for i in range(56)])
    picture, structure = stretch_structure()
    figure = plt.figure(figsize=(6.3, 6.3))
    ax = figure.add_axes([.15, .13, .81, .81])
    exported, values, handles = [], [], []
    for basis, label, color, linestyle, marker, width in SERIES:
        width *= LINE_SCALE
        selected = [by_key[(basis, i)] for i in range(56)]
        assert np.array_equal(distance, [r["distance_angstrom"] for r in selected])
        absolute = np.array([r["energy_hartree"] for r in selected])
        relative = np.array([r["relative_energy_kcal_mol"] for r in selected])
        assert np.allclose(relative, (absolute - absolute[0]) * HARTREE_KCAL_MOL, atol=1e-10, rtol=0)
        size = (4.2 if basis == "cc-pVTZ" else 3.3) * MARKER_SCALE
        ax.plot(distance, relative, color=color, ls=linestyle, lw=width,
                marker=marker, ms=size, markevery=2, markeredgewidth=.5, zorder=3)
        handles.append(Line2D([], [], color=color, ls=linestyle, lw=width,
                              marker=marker, ms=size, label=label))
        values.extend(relative)
        exported.extend(dict(basis=basis, index=r["index"], distance_angstrom=r["distance_angstrom"],
            energy_hartree=r["energy_hartree"], reference_energy_hartree=absolute[0],
            relative_energy_kcal_mol=float(y), accepted=True, nao=r["nao"])
            for r, y in zip(selected, relative))
    low, high = min(min(values), 0.), max(values)
    span = high - low
    bottom, top = low - .08 * span, high + (1.05 if PPT else .65) * span
    ax.set_ylim(bottom, top)
    ax.set_xlim(distance[0] - .10, distance[-1] + .12)
    ax.set_xticks([2., 3., 4., 5., 6., 7., 8.] if PPT
                  else [1.5, 2., 3., 4., 5., 6., 7., 8.])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel(r"C-C distance $R$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$E(R)-E(R_0)$ (kcal mol$^{-1}$)")
    ax.tick_params(top=True, right=True)
    ax.axhline(0., color="0.45", lw=.55 * LINE_SCALE, zorder=0)
    legend = ax.legend(handles=handles, loc="upper left",
                       handlelength=2., labelspacing=.3, borderaxespad=.65)
    width, height = .352, .352 * picture.height / picture.width
    if height > .288:
        height = .288; width = height * picture.width / picture.height
    box = [.97 - width, .97 - height, width, height]
    inset = ax.inset_axes(box); inset.imshow(picture); inset.set_axis_off()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    assert not legend.get_window_extent(renderer).overlaps(inset.get_window_extent(renderer))
    assert bottom + box[1] * (top - bottom) > max(values)
    stem = OUT / "ethane-cc-stretch"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(str(stem) + "." + extension, dpi=320)
    plt.close(figure)
    with (OUT / "plotted-points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader(); writer.writerows(exported)
    report = dict(molecule="ethane C2H6", data_package="2026-10-22", supersedes_data_package="2026-10-21 Case 05",
        style=style_metadata(),
        distance_points=56, curves=9, accepted_curve_points=504, mace_scf_converged=56,
        free_coefficient_converged=56, free_unconverged_distances_angstrom=[],
        reference_distance_angstrom=float(distance[0]), reference_policy="Each method uses its own E(R0)",
        hartree_to_kcal_mol=HARTREE_KCAL_MOL, available_methods=[r[0] for r in SERIES],
        figure_inches=[6.3, 6.3], png_dpi=320, inset_axes_box=box, structure=structure,
        source_sha256={str(p): sha(p) for p in [SOURCE, SUMMARY, TRADITIONAL, STRUCTURES,
            CASE / "structure-index.json", ROOT / "scripts/plot_ethane_cc_stretch.py"]},
        scope="Restricted closed-shell DF-RHF rigid stretching stress test. "
            "Not physical homolytic dissociation or a certified global RHF minimum. "
            "All 504 source points are accepted; historical failure markers and higher-branch energies are superseded. "
            "Relative energies are read directly from the corrected canonical table without smoothing.")
    (OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_sha256", "structure", "scope")}, indent=2))


if __name__ == "__main__":
    main()
