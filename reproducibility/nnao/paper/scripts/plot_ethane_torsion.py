"""Redraw the archived ethane torsion scan with a geometrically signed arrow.

Only archived structures and energies are read. No SCF, optimization, or
training is performed. The rotation arrow follows the source Rodrigues axis.
"""
from __future__ import annotations

import csv
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
import xyzrender
from xyzrender.export import svg_to_png_bytes
from plot_case08_reactions import (
    ROOT, PPT, OUTPUT_ROOT, LINE_SCALE, MARKER_SCALE,
    configure_style, sha, style_metadata,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PACKAGE = ROOT / "data/computational_examples/2026-10-21"
CASES = PACKAGE / "cases"
SOURCE = CASES / "06_ethane_fixed_polarization/artifacts/nnao-ethane-sto6g-polarized-20261020/all-curves.csv"
GEOMETRIES = CASES / "02_ethane_torsion/artifacts/nnao-ethane-torsion-20261018"
OUT = OUTPUT_ROOT / "ethane_torsion"
SERIES = [
    ("6-31G(d,p)", "6-31G(d,p)", "#8D933E", "-.", 1.7, None),
    ("cc-pVDZ", "cc-pVDZ", "#5C8C9E", ":", 1.7, None),
    ("cc-pVTZ", "cc-pVTZ", "#0057B8", "-", 2.8, "o"),
    ("NNAO / MACE", "NNAO/MACE", "#ef233c", "--", 1.8, "s"),
    ("NNAO / free contraction", "Free (44 AO)", "#171717", "--", 1.7, None),
    ("NNAO / C p2 / free", "Free + C-p (50 AO)", "#008879", "-.", 1.7, None),
    ("STO-6G + C(d), H(p)", "STO-6G + pol.", "#8A56A2", ":", 1.7, None),
]


def rotate(vector, axis, angle):
    return (vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)
            + np.outer(vector @ axis, axis) * (1 - np.cos(angle)))


def rotation_structure(angle, index):
    structure_file = GEOMETRIES / f"angle-{angle:03d}/structure.json"
    structure = json.loads(structure_file.read_text())
    xyz_file = PACKAGE / index[f"ethane-torsion/{angle:03d}"]["path"]
    before = sha(xyz_file)
    xyz = np.asarray(structure["coords_angstrom"])
    assert structure["symbols"].count("C") == 2 and structure["symbols"].count("H") == 6
    c0, c1 = structure["rotation_axis_atoms"]
    axis = xyz[c1] - xyz[c0]
    axis /= np.linalg.norm(axis)
    pivot = xyz[c1]
    tx, tz = np.deg2rad([42., -90.])
    rx = np.array([[1., 0., 0.], [0., np.cos(tx), -np.sin(tx)], [0., np.sin(tx), np.cos(tx)]])
    rz = np.array([[np.cos(tz), -np.sin(tz), 0.], [np.sin(tz), np.cos(tz), 0.], [0., 0., 1.]])
    camera = rz @ rx
    cc_display = (xyz[c1] - xyz[c0]) @ camera.T
    assert cc_display[0] > 0 and abs(cc_display[1]) < 1e-12
    molecule = xyzrender.load(xyz_file, charge=0, multiplicity=1, quick=True)
    for node, position in zip(molecule.graph, xyz @ camera.T):
        molecule.graph.nodes[node]["position"] = tuple(position)
    molecule.oriented = True
    config = xyzrender.build_config("paton", canvas_size=900, atom_scale=1.85,
                                   bond_width=12, atom_stroke_width=2.5)
    config.hide_h = config.auto_hide_h = False
    config.fixed_span, config.fixed_center = 6.4, (0., 0.)
    config.bond_orders = False
    svg = str(xyzrender.render(molecule, config=config, orient=False, hy=True))
    scale = (900 - 2 * config.padding) / config.fixed_span

    def project(points):
        p = np.atleast_2d(points) @ camera.T
        return np.column_stack((450 + scale * p[:, 0], 450 - scale * p[:, 1]))

    # Increasing theta follows exactly the source's +phi Rodrigues convention.
    theta = np.deg2rad(np.linspace(-150., -35., 90))
    arc = np.array([pivot + 1.72 * rotate(np.array([[1., 0., 0.]]), axis, t)[0] for t in theta])
    projected = project(arc)
    tangent = np.cross(axis, arc[0] - pivot) @ camera.T
    expected_screen_tangent = np.array([tangent[0], -tangent[1]])
    assert np.dot(projected[1] - projected[0], expected_screen_tangent) > 0
    path = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in projected)
    label_point = pivot + 2.12 * rotate(np.array([[1., 0., 0.]]), axis, np.deg2rad(-100.))[0]
    lx, ly = project(label_point)[0]
    arrow_width = 8 if PPT else 5
    font_weight = ' font-weight="bold"' if PPT else ""
    arrow = f'''<defs><marker id="torsion-positive-direction-head" markerWidth="20" markerHeight="16"
        refX="17" refY="8" orient="auto" markerUnits="userSpaceOnUse">
        <path d="M0,0 L20,8 L0,16 Z" fill="#4b5563"/></marker></defs>
        <path d="{path}" fill="none" stroke="#4b5563" stroke-width="{arrow_width}"
        stroke-linecap="round" marker-end="url(#torsion-positive-direction-head)"/>
        <text x="{lx:.3f}" y="{ly:.3f}" font-family="DejaVu Sans" font-size="42"{font_weight}
        text-anchor="middle" fill="#171717">+&#966;</text>'''
    svg = svg.replace("</svg>", arrow + "</svg>")
    stem = OUT / "structures" / f"ethane-{angle:03d}-rotation"
    Path(str(stem) + ".svg").write_text(svg)
    png = svg_to_png_bytes(svg, size=2700)
    Image.open(BytesIO(png)).save(str(stem) + ".png")
    image = Image.open(BytesIO(png)).convert("RGB")
    bounds = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    pad = 30
    box = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
    cropped = image.crop(box)
    assert before == sha(xyz_file)
    return cropped, dict(angle_deg=angle, source_xyz=str(xyz_file), source_xyz_sha256=before,
        source_structure=str(structure_file), source_structure_sha256=sha(structure_file),
        rotation_axis=axis.tolist(), pivot_angstrom=pivot.tolist(),
        rotating_atom_indices=structure["rotating_hydrogen_indices"],
        camera_rotation=camera.tolist(), arrow_theta_deg=[-150., -35.],
        cc_display_vector=cc_display.tolist(),
        projected_arrow_tangent_dot=float(np.dot(projected[1] - projected[0], expected_screen_tangent)),
        svg=str(Path(str(stem) + ".svg").relative_to(OUT)))


def main():
    (OUT / "structures").mkdir(parents=True, exist_ok=True)
    configure_style()
    rows = list(csv.DictReader(SOURCE.open()))
    data = {}
    for row in rows:
        row = dict(row)
        for k in ("angle_deg", "energy_hartree", "relative_energy_kcal_mol"):
            row[k] = float(row[k])
        row["nao"] = int(row["nao"])
        data[(row["basis"], row["angle_deg"])] = row
    assert len(rows) == len(data) == 275
    index = {r["label"]: r for r in json.loads((PACKAGE / "structure-index.json").read_text())
             if r["case"] == "02_ethane_torsion"}
    snapshots, structure_manifest = [], []
    for angle in (0, 60, 120):
        picture, metadata = rotation_structure(angle, index)
        snapshots.append(picture); structure_manifest.append(metadata)
    first = json.loads((GEOMETRIES / "angle-000/structure.json").read_text())
    fifth = json.loads((GEOMETRIES / "angle-005/structure.json").read_text())
    a = np.asarray(first["coords_angstrom"]); b = np.asarray(fifth["coords_angstrom"])
    moving = first["rotating_hydrogen_indices"]
    axis = np.array(structure_manifest[0]["rotation_axis"])
    predicted = rotate(a[moving] - a[1], axis, np.deg2rad(5)) + a[1]
    rotation_error = float(np.max(np.abs(predicted - b[moving])))
    assert rotation_error < 1e-12
    figure = plt.figure(figsize=(6.3, 6.3))
    ax = figure.add_axes([.15, .13, .81, .81])
    angles = list(range(0, 121, 5))
    exported, handles, values = [], [], []
    for basis, label, color, linestyle, linewidth, marker in SERIES:
        linewidth *= LINE_SCALE
        selected = [data[(basis, float(x))] for x in angles]
        assert len(selected) == 25 and selected[0]["relative_energy_kcal_mol"] == 0
        energy = [r["relative_energy_kcal_mol"] for r in selected]
        size = (4.2 if basis == "cc-pVTZ" else 3.5) * MARKER_SCALE
        ax.plot(angles, energy, color=color, ls=linestyle, lw=linewidth,
                marker=marker, ms=size, markevery=2, markeredgewidth=.5, zorder=3)
        handles.append(Line2D([], [], color=color, ls=linestyle, lw=linewidth,
                              marker=marker, ms=size, label=label))
        exported.extend(selected); values.extend(energy)
    low, high = min(min(values), 0.), max(values)
    span = high - low
    bottom, top = low - .07 * span, high + .80 * span
    ax.set_ylim(bottom, top)
    ax.set_xlim(-3, 123)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel(r"Methyl rotation $\phi$ (deg)")
    ax.set_ylabel(r"$E(\phi)-E(0)$ (kcal mol$^{-1}$)")
    ax.tick_params(top=True, right=True)
    ax.axhline(0., lw=.55 * LINE_SCALE, color="0.45", zorder=0)
    legend = ax.legend(handles=handles, loc="upper left",
                       handlelength=2., labelspacing=.3, borderaxespad=.65)
    width, height = .352, .352 * snapshots[0].height / snapshots[0].width
    if height > .288:
        height = .288; width = height * snapshots[0].width / snapshots[0].height
    box = [.97 - width, .97 - height, width, height]
    inset = ax.inset_axes(box); inset.imshow(snapshots[0]); inset.set_axis_off()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    assert not legend.get_window_extent(renderer).overlaps(inset.get_window_extent(renderer))
    assert bottom + box[1] * (top - bottom) > max(values)
    stem = OUT / "ethane-torsion-rotation"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(str(stem) + "." + extension, dpi=320)
    plt.close(figure)
    with (OUT / "plotted-points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader(); writer.writerows(exported)
    sheet = Image.new("RGB", (2100, 850), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 42)
    for i, (angle, picture) in enumerate(zip((0, 60, 120), snapshots)):
        picture = picture.copy(); picture.thumbnail((650, 720), Image.Resampling.LANCZOS)
        sheet.paste(picture, (i * 700 + (700 - picture.width) // 2, (750 - picture.height) // 2))
        draw.text((i * 700 + 350, 785), f"{angle} deg", fill="#171717", font=font, anchor="mm")
    sheet.save(OUT / "rotation-structures.png")
    report = dict(molecule="ethane", formula="C2H6", angles=angles,
        style=style_metadata(),
        selected_methods=[r[0] for r in SERIES], plotted_points=len(exported),
        source_sha256={str(p): sha(p) for p in [SOURCE, GEOMETRIES / "geometry.py", Path(__file__)]},
        figure_inches=[6.3, 6.3], png_dpi=320, legend_inside=True,
        rotation_0_to_5_coordinate_error_angstrom=rotation_error,
        positive_rotation="Source Rodrigues right-hand rotation about C0->C1; only H indices 5,6,7 move.",
        inset_axes_box=box, snapshots=structure_manifest,
        scope="Archived rigid ethane methyl rotation; each method uses its own 0-degree zero. "
              "50-AO expanded C-p results are free contractions only. STO-6G+pol. is a custom fixed control. "
              "The plus-phi arrow is projected from the same camera as the molecular structure.")
    (OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("snapshots", "source_sha256", "scope")}, indent=2))


if __name__ == "__main__":
    main()
