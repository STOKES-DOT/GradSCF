"""Draw structure-inset dimer curves in the approved case-08 figure style.

Reads the canonical archived values directly. Frozen-basis CP diagnostics are
written separately from the independently predicted monomer interaction curves.
Pass --ppt for the presentation version in output/ppt/.
"""
from __future__ import annotations

from collections import OrderedDict
import csv
from html import escape
import json
from pathlib import Path
import re

from plot_case08_reactions import (
    ROOT, CASE, DATASET, RESULTS, RENDERS, SERIES, configure_style,
    high_resolution_structure, sha, OUTPUT_ROOT, LINE_SCALE, MARKER_SCALE,
    style_metadata,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
from pypdf import PdfReader, PdfWriter


OUT = OUTPUT_ROOT / "case08_dimer_curves"
FLAT = RESULTS.parent / "interaction-cases.csv"
MODES = OrderedDict([
    ("model", ("model_interaction_kcal_mol", "Interaction energy")),
    ("frozen-cp", ("cp_interaction_kcal_mol", "CP interaction energy")),
])


def slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")


def draw_one(key, expected, rows, record, picture, mode):
    dataset, group = key
    field, ylabel = MODES[mode]
    scales = sorted(r["scale"] for r in expected)
    reference = {r["scale"]: r["reference_kcal_mol"] for r in expected}
    by_key = {(r["basis"], r["scale"]): r for r in rows}
    figure = plt.figure(figsize=(6.3, 6.3))
    ax = figure.add_axes([.15, .13, .81, .81])
    handles, values, exported = [], [], []
    for basis, label, color, linestyle, linewidth in SERIES:
        if basis == "Reference":
            energy = [reference[x] for x in scales]
            label = "Reference"
            marker, size = None, 0
        else:
            energy = [by_key[(basis, x)][field] for x in scales]
            marker, size = ("s", 3.6) if basis == "NNAO/MACE" else ("o", 2.6)
            if basis == "cc-pvtz":
                size = 4.2
        size *= MARKER_SCALE
        ax.plot(scales, energy, color=color, lw=linewidth, ls=linestyle,
                marker=marker, ms=size, markeredgewidth=.5 * LINE_SCALE, zorder=3)
        handles.append(Line2D([], [], color=color, lw=linewidth, ls=linestyle,
                              marker=marker, ms=size, label=label))
        values.extend(energy)
        exported.extend(dict(mode=mode, dataset=dataset, group=group, scale=x,
                             basis=basis, energy_kcal_mol=y) for x, y in zip(scales, energy))
    lowest, highest = min(min(values), 0.), max(max(values), 0.)
    span = highest - lowest
    lower, upper = lowest - .08 * span, highest + .60 * span
    ax.set_ylim(lower, upper)
    xspan = max(scales) - min(scales)
    ax.set_xlim(min(scales) - .025 * xspan, max(scales) + .025 * xspan)
    ax.set_xticks([.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Distance scale (dimensionless)")
    ax.set_ylabel(ylabel + r" (kcal mol$^{-1}$)")
    ax.tick_params(top=True, right=True)
    ax.axhline(0., color="0.45", lw=.55 * LINE_SCALE, zorder=0)
    legend = ax.legend(handles=handles, loc="upper left", ncol=1,
                       handlelength=2., labelspacing=.3, borderaxespad=.65)
    width = .44
    height = width * picture.height / picture.width
    if height > .31:
        height = .31
        width = height * picture.width / picture.height
    box = [.97 - width, .97 - height, width, height]
    inset = ax.inset_axes(box)
    inset.imshow(picture)
    inset.set_axis_off()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    assert not legend.get_window_extent(renderer).overlaps(inset.get_window_extent(renderer)), key
    image_lower_energy = lower + box[1] * (upper - lower)
    assert image_lower_energy > max(values), (key, mode)
    stem = f"{slug(dataset)}-{slug(group)}"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUT / mode / (stem + "." + extension), dpi=320)
    plt.close(figure)
    return exported, dict(mode=mode, dataset=dataset, group=group,
        group_title=record["group_title"], stem=stem, scales=scales,
        representative_structure=record["id"], representative_scale=record["scale"],
        source_svg=record["svg"], source_svg_sha256=record["svg_sha256"],
        inset_axes_box=box, inset_bottom_energy_kcal_mol=image_lower_energy,
        maximum_plotted_energy_kcal_mol=max(values), legend_location="upper left")


def write_index(groups, details):
    navigation, sections = [], []
    by_key = {(r["dataset"], r["group"]): r for r in details if r["mode"] == "model"}
    for key in groups:
        r = by_key[key]
        stem = r["stem"]
        title = f"{r['dataset']} | {r['group']}: {r['group_title']}"
        navigation.append(f'<a href="#{stem}">{escape(title)}</a>')
        versions = []
        for mode, heading in [("model", "Independent-monomer interaction"),
                              ("frozen-cp", "BSSE-corrected interaction (fixed dimer basis)")]:
            versions.append(f'<div><h3>{heading}</h3><a href="{mode}/{stem}.pdf">'
                f'<img src="{mode}/{stem}.png" alt="{escape(title)}; {heading}"></a><p>'
                + ' | '.join(f'<a href="{mode}/{stem}.{ext}">{ext.upper()}</a>'
                             for ext in ("pdf", "svg", "png")) + '</p></div>')
        sections.append(f'<section id="{stem}"><h2>{escape(title)}</h2>'
                        '<div class="versions">' + ''.join(versions) + '</div></section>')
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Case 08 dimer curves</title>'
        '<style>body{max-width:1450px;margin:30px auto;padding:0 20px;font:16px/1.5 Arial,sans-serif;color:#171717}'
        'img{width:100%;height:auto}.versions{display:grid;grid-template-columns:1fr 1fr;gap:25px}'
        'nav{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}section{margin-top:35px;border-top:1px solid #ddd}'
        'a{color:#175b9a}@media(max-width:850px){.versions,nav{grid-template-columns:1fr}}</style>'
        '<h1>Case 08 dimer interaction curves</h1>'
        '<p>Twelve dimers. Each molecular inset shows the archived distance-scale-1.00 structure. '
        'Lines connect the measured eight- or ten-point scans without adding interpolated data.</p>'
        '<p><a href="dimers-model-with-structures.pdf">12-page main-curve PDF</a> | '
        '<a href="dimers-frozen-cp-with-structures.pdf">12-page BSSE-corrected PDF</a> | '
        '<a href="overview-model.png">Main overview</a> | <a href="plotted-points.csv">Plotted data</a></p>'
        '<p>The main curves use independently evaluated isolated monomers. For the BSSE-corrected '
        'curves, the basis predicted for the dimer is held fixed while each monomer is solved '
        'with and without ghost functions. The SCF orbitals are optimized in each calculation; '
        'only the basis parameters are fixed. These curves use the recorded counterpoise '
        'energies and a common dimer auxiliary metric.</p>'
        '<nav>' + ''.join(navigation) + '</nav>' + ''.join(sections) + '</html>')


def main():
    configure_style()
    for mode in MODES:
        (OUT / mode).mkdir(parents=True, exist_ok=True)
    dataset = json.loads(DATASET.read_text())
    result = json.loads(RESULTS.read_text())
    manifest = json.loads((RENDERS / "manifest.json").read_text())
    rows = list(csv.DictReader(FLAT.open()))
    for row in rows:
        row["scale"] = float(row["scale"])
        for field in ("model_interaction_kcal_mol", "cp_interaction_kcal_mol", "bsse_kcal_mol", "reference_kcal_mol"):
            row[field] = float(row[field])
    assert len(rows) == 648
    assert len({(r["dataset"], r["group"], r["scale"], r["basis"]) for r in rows}) == 648
    raw = {(r["dataset"], r["group"], r["scale"], r["basis"]): r
           for r in result["results"] if r.get("task") == "curve"}
    for row in rows:
        actual = raw[(row["dataset"], row["group"], row["scale"], row["basis"])]
        assert actual["status"] == "passed"
        assert all(row[field] == actual[field] for field in
            ("model_interaction_kcal_mol", "cp_interaction_kcal_mol", "bsse_kcal_mol", "reference_kcal_mol"))
    groups = OrderedDict()
    for point in dataset["curves"]:
        groups.setdefault((point["dataset"], point["group"]), []).append(point)
    assert len(groups) == 12
    representatives = {(r["dataset"], r["group_id"]): r for r in manifest["records"]
                       if r["dataset"] != "BH9" and r["representative"]}
    points, details = [], []
    for key, expected in groups.items():
        record = representatives[key]
        assert record["scale"] == 1.0 and sha(RENDERS / record["svg"]) == record["svg_sha256"]
        picture = high_resolution_structure(record)
        selected = [r for r in rows if (r["dataset"], r["group"]) == key]
        for row in selected:
            assert row["reference_kcal_mol"] == next(r["reference_kcal_mol"] for r in expected if r["scale"] == row["scale"])
        for mode in MODES:
            exported, layout = draw_one(key, expected, selected, record, picture, mode)
            points.extend(exported); details.append(layout)
        print("Rendered", key[0], key[1], flush=True)
    with (OUT / "plotted-points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader(); writer.writerows(points)
    for mode in MODES:
        combined = PdfWriter()
        overview = Image.new("RGB", (2400, 3200), "white")
        for number, row in enumerate(r for r in details if r["mode"] == mode):
            combined.append(PdfReader(OUT / mode / (row["stem"] + ".pdf")))
            with Image.open(OUT / mode / (row["stem"] + ".png")) as image:
                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
                overview.paste(image, ((number % 3) * 800, (number // 3) * 800))
        combined.write(OUT / f"dimers-{mode}-with-structures.pdf")
        overview.save(OUT / f"overview-{mode}.png")
    write_index(groups, details)
    report = dict(dimers=12, main_figures=12, frozen_cp_figures=12,
        main_curve_points=sum(r["mode"] == "model" for r in points),
        frozen_cp_points=sum(r["mode"] == "frozen-cp" for r in points),
        plotted_methods=[r[0] for r in SERIES],
        source_sha256={str(p): sha(p) for p in [DATASET, RESULTS, FLAT, RENDERS / "manifest.json",
                                             Path(__file__), ROOT / "scripts/plot_case08_reactions.py"]},
        style=dict(**style_metadata(), figure_inches=[6.3, 6.3], png_dpi=320,
                   legend_inside=True, molecular_structure_inside=True,
                   triple_zeta_color="#0057B8", triple_zeta_width_pt=2.8 * LINE_SCALE,
                   triple_zeta_marker_pt=4.2 * MARKER_SCALE,
                   shared_style="plot_case08_reactions.configure_style / SERIES",
                   no_image_curve_overlap=True, no_legend_image_overlap=True),
        figures=details,
        scope="Archived DF-HF comparisons; dimensionless source distance scales. "
            "Scale-1.00 molecular views; no new geometries or electronic calculations. "
            "Reference differences include electronic-method and basis effects. "
            "Frozen CP values are not model interaction energies plus BSSE.")
    (OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("figures", "source_sha256", "scope")}, indent=2))


if __name__ == "__main__":
    main()
