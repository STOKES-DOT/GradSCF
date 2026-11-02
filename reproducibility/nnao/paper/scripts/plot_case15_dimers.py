"""Redraw corrected, isolated-reference no-CP dimer curves; use --ppt.

The fixed-output MACE control is intentionally omitted from these figures.
The Case 15 source archive is read without modification.
"""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path
import zipfile

from plot_case08_reactions import (
    ROOT, OUTPUT_ROOT, RENDERS, LINE_SCALE, MARKER_SCALE,
    configure_style, high_resolution_structure, sha, style_metadata,
)
from plot_case08_dimers import slug
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
from pypdf import PdfReader, PdfWriter


CASE = ROOT / "data/computational_examples/2026-10-26/cases/15_dimer_cp_output_controls"
SOURCE = CASE / "combined/curves.csv"
DEFINITIONS = CASE / "inputs/cases.json"
STRUCTURES = CASE / "representative-structures/index.json"
OUT = OUTPUT_ROOT / "case15_dimer_controls"
SERIES = [
    ("Reference", "Reference", "#171717", "-", 1.7, None),
    ("sto-6g", "STO-6G", "#89939c", "-", 1.3, "o"),
    ("STO-6G+P", "STO-6G+P", "#8A56A2", ":", 1.7, "D"),
    ("6-31g(d,p)", "6-31G(d,p)", "#8D933E", "-.", 1.7, "o"),
    ("def2-svp", "def2-SVP", "#ba718b", "--", 1.7, "o"),
    ("cc-pvdz", "cc-pVDZ", "#5C8C9E", ":", 1.7, "o"),
    ("cc-pvtz", "cc-pVTZ", "#0057B8", "-", 2.8, "o"),
    ("NNAO/MACE", "NNAO/MACE", "#ef233c", "--", 1.8, "s"),
]


def draw_one(key, points, rows, record, picture):
    x = [point["scale"] for point in points]
    source = {(r["basis"], float(r["scale"])): r for r in rows}
    figure = plt.figure(figsize=(6.3, 6.3))
    ax = figure.add_axes([.15, .13, .81, .81])
    handles, values, exported = [], [], []
    for basis, label, color, linestyle, width, marker in SERIES:
        energy = ([point["reference_kcal_mol"] for point in points] if basis == "Reference"
                  else [float(source[basis, scale]["interaction_kcal_mol"]) for scale in x])
        size = (4.2 if basis == "cc-pvtz" else 3.6 if basis == "NNAO/MACE" else 2.6) * MARKER_SCALE
        ax.plot(x, energy, color=color, ls=linestyle, lw=width * LINE_SCALE,
                marker=marker, ms=size, markeredgewidth=.5 * LINE_SCALE, zorder=3)
        handles.append(Line2D([], [], color=color, ls=linestyle, lw=width * LINE_SCALE,
                              marker=marker, ms=size, label=label))
        values.extend(energy)
        exported.extend(dict(dataset=key[0], group=key[1], cp_mode="no_cp", basis=basis,
                             scale=scale, interaction_kcal_mol=float(y)) for scale, y in zip(x, energy))
    low, high = min(min(values), 0.), max(max(values), 0.)
    span = max(high - low, 1e-3)
    lower, upper = low - .08 * span, high + .65 * span
    margin = .025 * (max(x) - min(x))
    ax.set_xlim(min(x) - margin, max(x) + margin)
    ax.set_xticks([.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Distance scale (dimensionless)")
    ax.set_ylabel(r"Interaction energy (kcal mol$^{-1}$)")
    ax.tick_params(top=True, right=True)
    ax.axhline(0., color=".45", lw=.55 * LINE_SCALE, zorder=0)
    legend = ax.legend(handles=handles, loc="upper left", handlelength=1.8,
                       labelspacing=.3, borderaxespad=.55)
    width = .44
    height = width * picture.height / picture.width
    if height > .31:
        height = .31
        width = height * picture.width / picture.height
    box = [.97 - width, .97 - height, width, height]
    inset = ax.inset_axes(box)
    inset.imshow(picture)
    inset.set_axis_off()
    for attempt in range(15):
        ax.set_ylim(lower, upper)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer)
        image_box = inset.get_window_extent(renderer)
        assert not legend_box.overlaps(image_box), key
        legend_low = ax.transData.inverted().transform((0., legend_box.y0))[1]
        image_low = lower + box[1] * (upper - lower)
        if min(legend_low, image_low) > max(values):
            break
        upper += .12 * span
    assert min(legend_low, image_low) > max(values), key
    stem = slug("-".join(key))
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUT / "no-cp" / (stem + "." + extension), dpi=320)
    plt.close(figure)
    return exported, dict(dataset=key[0], group=key[1], stem=stem,
                         title=record["group_title"], representative_structure=record["id"],
                         source_svg_sha256=record["svg_sha256"], y_limits=[lower, upper],
                         inset_axes_box=box, no_legend_image_overlap=True,
                         no_legend_data_overlap=True, no_image_data_overlap=True)


def main():
    configure_style()
    (OUT / "no-cp").mkdir(parents=True, exist_ok=True)
    with SOURCE.open() as stream:
        all_rows = list(csv.DictReader(stream))
    assert len(all_rows) == 1728
    rows = [r for r in all_rows if r["cp_mode"] == "no_cp" and r["basis"] != "NNAO/MACE-fixed"]
    assert len(rows) == 756
    wanted = {item[0] for item in SERIES} - {"Reference"}
    assert {r["basis"] for r in rows} == wanted
    definitions = json.loads(DEFINITIONS.read_text())["definitions"]
    assert len(definitions) == 108
    renders = {r["id"]: r for r in json.loads(STRUCTURES.read_text())["records"]}
    original_plot = json.loads((CASE / "figures/plot-values.json").read_text())
    original_values = {(r["dataset"], r["group"], r["basis"], r["scale"]): r["value_kcal_mol"]
                       for r in original_plot if r["cp_mode"] == "no_cp" and r["basis"] != "NNAO/MACE-fixed"}
    for row in rows:
        assert float(row["interaction_kcal_mol"]) == original_values[
            row["dataset"], row["group"], row["basis"], float(row["scale"])]
    hbr = next(r for r in rows if r["dataset"] == "HB300SPXx10" and r["group"] == "1.28"
               and r["basis"] == "NNAO/MACE" and float(r["scale"]) == .8)
    assert abs(float(hbr["interaction_kcal_mol"]) - 1.0793349747841636) < 1e-12
    points, figures = [], []
    for key in sorted({(r["dataset"], r["group"]) for r in definitions}):
        selected = [r for r in rows if (r["dataset"], r["group"]) == key]
        expected = sorted((r for r in definitions if (r["dataset"], r["group"]) == key), key=lambda r: r["scale"])
        anchor = next(r for r in expected if r["scale"] == 1.)
        record = renders[anchor["identifier"]]
        assert record["scale"] == 1. and sha(RENDERS / record["svg"]) == record["svg_sha256"]
        for basis in wanted:
            basis_rows = [r for r in selected if r["basis"] == basis]
            assert len(basis_rows) == len(expected)
            assert len({(r["isolated_a_hartree"], r["isolated_b_hartree"]) for r in basis_rows}) == 1
            assert {float(r["scale"]) for r in basis_rows} == {r["scale"] for r in expected}
        exported, metadata = draw_one(key, expected, selected, record, high_resolution_structure(record))
        for row in exported:
            assert row["interaction_kcal_mol"] == original_values[
                row["dataset"], row["group"], row["basis"], row["scale"]]
        points.extend(exported)
        figures.append(metadata)
        print("Rendered", *key, flush=True)
    assert len(figures) == 12 and len(points) == 864
    with (OUT / "plotted-no-cp-values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    book = PdfWriter()
    overview = Image.new("RGB", (2400, 3200), "white")
    cards = []
    with zipfile.ZipFile(OUT / "dimers-no-cp-pngs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for number, item in enumerate(figures):
            stem = item["stem"]
            book.append(PdfReader(OUT / "no-cp" / (stem + ".pdf")))
            archive.write(OUT / "no-cp" / (stem + ".png"), stem + ".png")
            with Image.open(OUT / "no-cp" / (stem + ".png")) as image:
                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
                overview.paste(image, ((number % 3) * 800, (number // 3) * 800))
            title = item["dataset"] + " | " + item["group"] + ": " + item["title"]
            cards.append('<section><h2>' + escape(title) + '</h2><a href="no-cp/' + stem + '.pdf">'
                '<img src="no-cp/' + stem + '.png" alt="' + escape(title) + '"></a><p>' +
                ' | '.join('<a href="no-cp/' + stem + '.' + ext + '">' + ext.upper() + '</a>'
                           for ext in ("png", "pdf", "svg")) + '</p></section>')
    book.write(OUT / "dimers-no-cp-with-structures.pdf")
    overview.save(OUT / "overview-no-cp.png")
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Corrected dimer interactions</title>'
        '<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.5 Arial;color:#171717}'
        'img{width:100%;height:auto}section{margin:35px 0}a{color:#0057b8}</style>'
        '<h1>Corrected dimer interactions: no CP</h1><p>12 dimers, with adaptive MACE, six conventional bases '
        'and the original reference curves. The fixed-output MACE control is omitted.</p>'
        '<p>Each scan uses constant independently evaluated isolated monomer energies. '
        'No CP correction, vertical shift, clipping or forced asymptotic zero is applied. '
        'Insets show the original distance-scale-1.00 structures.</p>'
        '<p><a href="dimers-no-cp-with-structures.pdf">12-page PDF</a> | '
        '<a href="dimers-no-cp-pngs.zip">All PNGs</a> | <a href="overview-no-cp.png">Overview</a></p>'
        + ''.join(cards) + '</html>')
    report = dict(figures=figures, plotted_values=len(points), calculated_values=756, reference_values=108,
        cp_mode="no_cp", mace_policy="adaptive only", excluded_basis="NNAO/MACE-fixed",
        source_correction_date="2026-10-27", style=dict(**style_metadata(), figure_inches=[6.3, 6.3],
            png_dpi=320, cc_pvtz_color="#0057B8", cc_pvtz_width_pt=2.8 * LINE_SCALE),
        source_sha256={str(p): sha(p) for p in [SOURCE, DEFINITIONS, STRUCTURES,
            CASE / "figures/plot-values.json", CASE / "figures/dimers-no-cp-with-structures.pdf",
            CASE / "isolated-reference-map.json", CASE / "README.md", Path(__file__),
            ROOT / "scripts/plot_case08_reactions.py"]},
        formula="E_AB(R) - E_A(isolated) - E_B(isolated), with constant references within each scan",
        scope="Archived corrected no-CP interactions only; no SCF or model inference. "
              "Adaptive MACE and six conventional bases are shown; fixed-output MACE is excluded. "
              "Reference curves include electronic-method and basis differences.")
    (OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("figures", "source_sha256", "scope")}, indent=2))


if __name__ == "__main__":
    main()
