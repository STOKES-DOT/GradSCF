"""Render archived BSSE analyses; use --ppt for the presentation profile.

Reads the 2026-10-26 supplement without changing its tables, frozen bases or
figures. No electronic-structure calculations are performed.
"""
from __future__ import annotations

from collections import OrderedDict
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
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import LogLocator
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter


CASE = ROOT / "data/computational_examples/2026-10-26/cases/14_bsse_counterpoise"
OUT = OUTPUT_ROOT / "bsse_counterpoise"
TABLE = CASE / "tables/distance-counterpoise-all-cases.csv"
SUMMARY = CASE / "tables/s22-summary.csv"
SERIES = [
    ("sto-6g", "STO-6G", "#9A8268", "-.", 1.7),
    ("STO-6G+P", "STO-6G+P", "#8A56A2", ":", 1.7),
    ("6-31g(d,p)", "6-31G(d,p)", "#8D933E", "-.", 1.7),
    ("def2-svp", "def2-SVP", "#ba718b", "-.", 1.7),
    ("cc-pvdz", "cc-pVDZ", "#5C8C9E", ":", 1.7),
    ("cc-pvtz", "cc-pVTZ", "#0057B8", "-", 2.8),
    ("ma-def2-qzvpp", "ma-def2-QZVPP", "#343434", "-", 1.7),
    ("NNAO/MACE", "NNAO/MACE", "#ef233c", "--", 1.8),
]
SERIES = [(basis, label, color, linestyle, width * LINE_SCALE)
          for basis, label, color, linestyle, width in SERIES]


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def curve_clear(lines, box):
    """Check the displayed polylines, including log-axis transforms."""
    padded = Bbox.from_extents(box.x0 - 4, box.y0 - 4, box.x1 + 4, box.y1 + 4)
    return all(not PlotPath(line.get_transform().transform(line.get_xydata())).intersects_bbox(
        padded, filled=False) for line in lines)


def assert_text_inside(figure):
    renderer = figure.canvas.get_renderer()
    omitted_ticks = set()
    for ax in figure.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not min(limits) <= tick.get_loc() <= max(limits):
                    omitted_ticks.update((tick.label1, tick.label2))
    for text in figure.findobj(plt.Text):
        if text not in omitted_ticks and text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert figure.bbox.contains(box.x0, box.y0) and figure.bbox.contains(box.x1, box.y1), text.get_text()


def draw_distance(key, rows, picture, record):
    figure = plt.figure(figsize=(6.3, 6.3))
    ax = figure.add_axes([.17, .13, .79, .81])
    handles, lines, plotted, values = [], [], [], []
    for basis, label, color, linestyle, linewidth in SERIES:
        selected = sorted((r for r in rows if r["basis"] == basis), key=lambda r: float(r["scale"]))
        if not selected:
            continue
        scales = [float(r["scale"]) for r in selected]
        energy = [float(r["bsse_kcal_mol"]) for r in selected]
        assert all(value > 0 for value in energy)
        marker = "s" if basis == "NNAO/MACE" else "o"
        size = (4.2 if basis == "cc-pvtz" else 3.5) * MARKER_SCALE
        line, = ax.plot(scales, energy, color=color, ls=linestyle, lw=linewidth,
                        marker=marker, ms=size, markeredgewidth=.5 * LINE_SCALE, zorder=3)
        lines.append(line)
        handles.append(Line2D([], [], color=color, ls=linestyle, lw=linewidth,
                              marker=marker, ms=size, label=label))
        values.extend(energy)
        plotted.extend(dict(dataset=key[0], group=key[1], identifier=r["identifier"],
                            basis=basis, scale=float(r["scale"]),
                            value_kcal_mol=float(r["bsse_kcal_mol"])) for r in selected)
    assert len(lines) == 6
    ax.set_yscale("log")
    ax.set_xlabel("Distance scale (dimensionless)")
    ax.set_ylabel(r"BSSE (kcal mol$^{-1}$)")
    ax.set_xlim(.77, 2.03)
    ax.set_xticks([.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2., 5.), numticks=100))
    ax.tick_params(which="both", top=True, right=True)
    ax.tick_params(which="minor", length=2.5, width=.8)
    width = .44
    height = width * picture.height / picture.width
    if height > .31:
        height = .31
        width = height * picture.width / picture.height
    box = [.97 - width, .97 - height, width, height]
    inset = ax.inset_axes(box)
    inset.imshow(picture)
    inset.set_axis_off()
    low, high = np.log10([min(values), max(values)])
    span = high - low
    lower = low - .08 * span
    upper = high + .12 * span
    legend = None
    for attempt in range(20):
        ax.set_ylim(10 ** lower, 10 ** upper)
        for location in ("lower left", "upper left"):
            if legend is not None:
                legend.remove()
            legend = ax.legend(handles=handles, loc=location, handlelength=2.,
                               labelspacing=.3, borderaxespad=.65)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            legend_box = legend.get_window_extent(renderer)
            image_box = inset.get_window_extent(renderer)
            clear = (not legend_box.overlaps(image_box)
                     and curve_clear(lines, legend_box) and curve_clear(lines, image_box))
            if clear:
                break
        if clear:
            break
        upper += .12 * span
    assert clear, key
    assert_text_inside(figure)
    stem = "distance-bsse/" + slug("-".join(key))
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUT / (stem + "." + extension), dpi=320)
    plt.close(figure)
    return plotted, dict(stem=stem, title=record["group_title"], dataset=key[0], group=key[1],
                         legend_location=location, y_scale="log", y_limits=[10 ** lower, 10 ** upper],
                         inset_axes_box=box, no_curve_image_overlap=True, no_curve_legend_overlap=True,
                         representative_structure=record["id"], representative_scale=1.,
                         source_svg_sha256=record["svg_sha256"])


def write_gallery(figures):
    sections = []
    for title, selected in [("S22 basis comparisons", figures[:3]),
                            ("BSSE distance curves", figures[3:])]:
        cards = []
        for item in selected:
            stem = item["stem"]
            cards.append('<article><h3>' + escape(item["title"]) + '</h3><a href="' + stem + '.png">'
                '<img src="' + stem + '.png" alt="' + escape(item["title"]) + '"></a><p>' +
                ' | '.join('<a href="' + stem + '.' + ext + '">' + ext.upper() + '</a>'
                           for ext in ("png", "pdf", "svg")) + '</p></article>')
        sections.append('<section><h2>' + title + '</h2><div class="grid">' + ''.join(cards) + '</div></section>')
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>BSSE presentation figures</title>'
        '<style>body{max-width:1450px;margin:30px auto;padding:0 20px;font:16px/1.5 Arial;color:#171717}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:25px}'
        'img{width:100%;height:auto}h3{font-size:16px}a{color:#0057b8}</style>'
        '<h1>BSSE presentation figures</h1><p><a href="bsse-ppt-pngs.zip">All PNGs</a> | '
        '<a href="bsse-all-figures.pdf">Combined PDF</a> | <a href="verification.json">Verification</a></p>'
        '<p>S22 full-22 and common-12 cohorts are shown separately. Distance plots use logarithmic y axes; '
        'all values are positive. Insets show distance scale 1.00. BSSE uses fixed dimer-predicted '
        'basis parameters and the common dimer auxiliary metric.</p>'
        '<p>The STO-6G means retain the large archived S22/12 value (282.1534 kcal/mol). '
        'Its asymmetric own-fragment energies warrant a separate SCF-branch review; '
        'the means should be interpreted with this source limitation. '
        '<a href="README.md">Source notes</a></p>' + ''.join(sections) + '</html>')


def main():
    configure_style()
    (OUT / "distance-bsse").mkdir(parents=True, exist_ok=True)
    rows = read_csv(TABLE)
    assert len(rows) == 648 and all(r["status"] == "passed" for r in rows)
    old_values = json.loads((CASE / "plot-values.json").read_text())
    expected = {(r["identifier"], r["basis"]): r["value_kcal_mol"]
                for r in old_values if r.get("identifier")}
    for row in rows:
        assert float(row["bsse_kcal_mol"]) == expected[(row["identifier"], row["basis"])]
        assert abs(float(row["bsse_kcal_mol"]) - (float(row["cp_interaction_kcal_mol"])
                   - float(row["uncorrected_interaction_kcal_mol"]))) < 5e-10
    manifest = json.loads((RENDERS / "manifest.json").read_text())
    representatives = {(r["dataset"], r["group_id"]): r for r in manifest["records"]
                       if r["dataset"] != "BH9" and r["representative"]}
    structure_index = json.loads((CASE / "structure-index.json").read_text())
    groups = OrderedDict()
    for row in rows:
        groups.setdefault((row["dataset"], row["group"]), []).append(row)
    assert len(groups) == 12
    points, figures = [], []
    for key, selected in groups.items():
        record = representatives[key]
        current = next(r for r in structure_index if r["identifier"] == record["id"])
        original_lines = Path(record["source_xyz"]).read_text().splitlines()[2:]
        original = [(line.split()[0], list(map(float, line.split()[1:4])))
                    for line in original_lines if line.strip()]
        assert [symbol for symbol, _ in original] == current["symbols"]
        assert np.array_equal([coords for _, coords in original], current["coords_angstrom"])
        assert sha(RENDERS / record["svg"]) == record["svg_sha256"]
        plotted, layout = draw_distance(key, selected, high_resolution_structure(record), record)
        points.extend(plotted)
        figures.append(layout)
        print("Rendered", *key, flush=True)
    from plot_bsse_s22 import draw_s22
    summary_points, summary_figures = draw_s22(CASE, OUT, SERIES)
    figures = list(summary_figures) + figures
    assert len(points) == 648 and len(summary_points) == 15
    (OUT / "plotted-values.json").write_text(json.dumps(summary_points + points, indent=2) + "\n")
    book = PdfWriter()
    with zipfile.ZipFile(OUT / "bsse-ppt-pngs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for item in figures:
            stem = item["stem"]
            archive.write(OUT / (stem + ".png"), stem + ".png")
            book.append(PdfReader(OUT / (stem + ".pdf")))
    book.write(OUT / "bsse-all-figures.pdf")
    sheet = Image.new("RGB", (2400, 3200), "white")
    for number, item in enumerate(figures[3:]):
        with Image.open(OUT / (item["stem"] + ".png")) as image:
            image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            sheet.paste(image, ((number % 3) * 800, (number // 3) * 800))
    sheet.save(OUT / "distance-bsse-overview.png")
    write_gallery(figures)
    report = dict(figures=figures, distance_systems=12, distance_bsse_points=len(points),
                  s22_cohort_means=len(summary_points), style=style_metadata(),
                  source_sha256={str(p): sha(p) for p in [TABLE, SUMMARY, CASE / "plot-values.json",
                      CASE / "structure-index.json", CASE / "model-provenance.json", Path(__file__),
                      ROOT / "scripts/plot_bsse_s22.py", ROOT / "scripts/plot_case08_reactions.py"]},
                  source_caveats=["The archived S22/12 STO-6G BSSE is 282.1533576987432 kcal/mol. "
                      "Its asymmetric own-fragment energies can reflect different SCF branches; "
                      "no branch diagnosis or recalculation is performed here. Means retain this record.",
                      "STO-6G+P uses def2-SVP polarization shells and is distinct from the ethane fixed control."],
                  scope="Archived frozen-dimer-basis BSSE only. Separate full-22 and common-12 S22 cohorts. "
                        "Ten ma-def2-QZVPP failures are excluded from full-22 and never replaced by zero. "
                        "Distance markers are the original samples; connecting lines add no calculated points. "
                        "No electronic-structure calculations or raw data modifications.")
    (OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("figures", "source_sha256", "scope")}, indent=2))


if __name__ == "__main__":
    main()
