"""Plot the complete non-QZ Case 18 snapshot using the established house style.

Read saved summary tables only. No inference, SCF, optimization, or training.
Use --ppt for bold presentation typography in output/ppt/.
"""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path
import shutil
import zipfile

from plot_case08_reactions import ROOT, OUTPUT_ROOT, PPT, configure_style, sha, style_metadata
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
import fitz


CASE = ROOT / "data/computational_examples/2026-10-29/cases/18_basis_accuracy_partial_non_qz"
SOURCE = CASE / "tables/summary.csv"
OUT = OUTPUT_ROOT / "basis_accuracy_benchmark"
METHODS = [("mp2", "DF-MP2 | frozen core"), ("wb97m-v", r"$\omega$B97M-V | grid level 3")]
COLORS = {"STO": "#89939c", "Pople": "#8D933E", "def2": "#ba718b",
          "cc": "#5C8C9E", "NNAO": "#ef233c"}
FIGURE_INCHES = (8.8, 7.6)
XLIMITS = (0., 195.)
YLIMITS = (.28, 10.)
SIZE_SCALE = 1.4 ** 2 if PPT else 1.


def read_rows():
    with SOURCE.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48 and len({(r["basis_key"], r["method"]) for r in rows}) == 48
    for row in rows:
        assert (int(row["structures"]), int(row["isomerizations"]), int(row["dimers"])) == (30, 6, 6)
        for key in ("mean_ao", "mean_primitive", "balanced_mae_kcal_mol",
                    "isomer_mae_kcal_mol", "dimer_mae_kcal_mol"):
            row[key] = float(row[key])
            assert np.isfinite(row[key]) and row[key] > 0
        assert abs(row["balanced_mae_kcal_mol"] - .5 *
                   (row["isomer_mae_kcal_mol"] + row["dimer_mae_kcal_mol"])) < 1e-12
    for key in {r["basis_key"] for r in rows}:
        pair = [r for r in rows if r["basis_key"] == key]
        assert len(pair) == 2
        assert pair[0]["mean_ao"] == pair[1]["mean_ao"]
        assert pair[0]["mean_primitive"] == pair[1]["mean_primitive"]
    return rows


def point_color(row):
    return "#0057B8" if row["label"] == "cc-pVTZ" else COLORS[row["family"]]


def place_labels(ax, rows, blocked):
    """Find clear positions for all names without changing any data coordinate."""
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    factor = figure.dpi / 72.
    fontsize = 12. if PPT else 9.5
    font = FontProperties(family="DejaVu Sans", weight="bold" if PPT else "normal", size=fontsize)
    limits = ax.bbox.padded(-4 * factor)
    markers, marker_keys = [], []
    for row in rows:
        px, py = ax.transData.transform((row["mean_ao"], row["balanced_mae_kcal_mol"]))
        radius = .5 * np.sqrt(SIZE_SCALE * row["mean_primitive"]) * factor + 2 * factor
        markers.append(Bbox.from_extents(px - radius, py - radius, px + radius, py + radius))
        marker_keys.append(row["basis_key"])
    occupied = list(blocked)
    annotations, records, leaders = [], [], []

    def clear_candidate(box, point):
        if any(box.overlaps(other) for other in occupied + markers):
            return None
        if any(PlotPath(segment).intersects_bbox(box, filled=False) for segment in leaders):
            return None
        end = np.clip(point, [box.x0, box.y0], [box.x1, box.y1])
        direction = end - point
        radius = .5 * np.sqrt(SIZE_SCALE * row["mean_primitive"]) * factor + factor
        start = point + direction / np.linalg.norm(direction) * radius
        segment = np.vstack([start, end])
        path = PlotPath(segment)
        if any(path.intersects_bbox(other, filled=False) for other in occupied):
            return None
        # Nearby overlapping circles are intrinsic to the data, but a leader
        # must not pass through a separate, unrelated point on its way out.
        if any(path.intersects_bbox(other, filled=False)
               for key, other in zip(marker_keys, markers)
               if key != row["basis_key"] and not other.contains(*point)):
            return None
        crossings = sum(path.intersects_path(PlotPath(other), filled=False) for other in leaders)
        return segment, crossings

    order = sorted(rows, key=lambda row: (row["kind"] != "nnao", row["balanced_mae_kcal_mol"]))
    for row in order:
        name = row["label"].replace("NNAO-", "")
        width, height, descent = renderer.get_text_width_height_descent(name, font, False)
        width += 5 * factor
        height = max(height + descent + 4 * factor, fontsize * factor * 1.25)
        point = ax.transData.transform((row["mean_ao"], row["balanced_mae_kcal_mol"]))
        candidates = []
        for dx in (12, 22, 36, 52, 72, 98, 130, -12, -22, -36, -52, -72, -98, -130):
            for dy in (0, 12, -12, 24, -24, 38, -38, 55, -55, 76, -76, 100, -100):
                left = point[0] + dx * factor - (width if dx < 0 else 0.)
                bottom = point[1] + dy * factor - height / 2
                box = Bbox.from_bounds(left, bottom, width, height)
                if (box.x0 < limits.x0 or box.x1 > limits.x1 or
                        box.y0 < limits.y0 or box.y1 > limits.y1):
                    continue
                clear = clear_candidate(box, point)
                if clear is None:
                    continue
                # Short, mostly horizontal leaders make point/name pairing easy.
                segment, crossings = clear
                score = dx * dx + 1.3 * dy * dy + (35 if dx < 0 else 0) + crossings * 50000
                candidates.append((score, box, segment))
        if not candidates:
            # A bounded display-space search handles dense same-AO clusters.
            for left in np.arange(limits.x0, limits.x1 - width, 4 * factor):
                for bottom in np.arange(limits.y0, limits.y1 - height, 5 * factor):
                    box = Bbox.from_bounds(left, bottom, width, height)
                    clear = clear_candidate(box, point)
                    if clear is None:
                        continue
                    segment, crossings = clear
                    distance = np.sum((np.array([left + width / 2, bottom + height / 2]) - point) ** 2)
                    candidates.append((distance / factor ** 2 + crossings * 50000, box, segment))
        assert candidates, (row["method"], name)
        _, box, segment = min(candidates, key=lambda item: item[0])
        position = ax.transData.inverted().transform((box.x0 + 2.5 * factor, .5 * (box.y0 + box.y1)))
        data_segment = ax.transData.inverted().transform(segment)
        ax.plot(data_segment[:, 0], data_segment[:, 1], color="#7d858c",
                lw=.7 if PPT else .5, solid_capstyle="butt", zorder=2)
        annotation = ax.text(*position, name, ha="left", va="center", fontsize=fontsize,
            fontweight="bold" if PPT else "normal",
            color="#cf1a2e" if row["kind"] == "nnao" else point_color(row) if row["label"] == "cc-pVTZ" else "#171717",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.92, pad=.10),
            zorder=6)
        annotations.append(annotation)
        occupied.append(box)
        leaders.append(segment)
        records.append(dict(basis_key=row["basis_key"], text=name,
                            point=[row["mean_ao"], row["balanced_mae_kcal_mol"]],
                            text_position=position.tolist(), leader_data_coordinates=data_segment.tolist()))
    figure.canvas.draw()
    actual = [text.get_bbox_patch().get_window_extent(renderer).padded(.5 * factor) for text in annotations]
    for i, box in enumerate(actual):
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
        assert not any(box.overlaps(other) for other in actual[i + 1:]), (records[i], "labels overlap")
        assert not any(box.overlaps(other) for other in blocked), (records[i], "legend overlap")
        assert not any(box.overlaps(other) for other in markers), (records[i], "bubble overlap")
        assert not any(PlotPath(segment).intersects_bbox(box, filled=False)
                       for j, segment in enumerate(leaders) if j != i), (records[i], "leader crosses label")
    return records


def draw_plot(rows, method, title):
    selected = [r for r in rows if r["method"] == method]
    assert len(selected) == 24
    figure = plt.figure(figsize=FIGURE_INCHES)
    ax = figure.add_axes([.13, .17, .84, .74])
    ax.set(xlim=XLIMITS, ylim=YLIMITS, yscale="log",
           xlabel="Mean contracted spherical AOs", ylabel=r"MAE vs database (kcal mol$^{-1}$)")
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.yaxis.set_major_locator(FixedLocator([.3, .5, 1., 2., 5., 10.]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2., 3., 4., 6., 7., 8., 9.)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", top=True, right=True)
    ax.tick_params(which="minor", length=2.5, width=.8)
    ax.grid(False)
    for row in sorted(selected, key=lambda item: -item["mean_primitive"]):
        emphasized = row["kind"] == "nnao" or row["label"] == "cc-pVTZ"
        ax.scatter(row["mean_ao"], row["balanced_mae_kcal_mol"],
                   s=SIZE_SCALE * row["mean_primitive"], marker="o", color=point_color(row),
                   edgecolors="#171717" if row["kind"] == "nnao" else point_color(row) if emphasized else "white",
                   linewidths=1.1 if emphasized else .7, alpha=.9, zorder=3)
    handles = [Line2D([], [], marker="o", ls="", ms=7.5, color=color,
                      markeredgecolor="#171717" if family == "NNAO" else "white", label=family)
               for family, color in COLORS.items()]
    family_legend = ax.legend(handles=[handles[i] for i in (0, 3, 1, 4, 2)], loc="upper left", ncol=3,
                              fontsize=11.5 if PPT else 9., handlelength=1.,
                              handletextpad=.35, columnspacing=1., borderaxespad=.5,
                              labelspacing=.45, frameon=False)
    ax.add_artist(family_legend)
    size_handles = [ax.scatter([], [], s=SIZE_SCALE * size, color="#b0b7bd",
                               edgecolors="white", lw=.7, label=str(size)) for size in (80, 130, 180)]
    size_legend = ax.legend(handles=size_handles, title="Mean primitive GTOs", loc="upper right",
                           fontsize=11. if PPT else 9., title_fontsize=11. if PPT else 9.,
                           labelspacing=.8, borderaxespad=.6, handletextpad=.7, frameon=False)
    figure.text(.13, .972, title, fontsize=18 if PPT else 14, fontweight="bold", va="top")
    figure.text(.13, .055, "24 non-QZ bases | 30 structures | 6 isomerizations + 6 dimers | no CP",
                fontsize=10. if PPT else 8., color="#333333", va="center")
    figure.text(.13, .025, "References: ISO34 — W1-F12; S22B — estimated CCSD(T)/CBS",
                fontsize=10. if PPT else 8., color="#333333", va="center")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    blocked = [legend.get_window_extent(renderer).padded(3) for legend in (family_legend, size_legend)]
    labels = place_labels(ax, selected, blocked)
    stem = f"basis-accuracy-{method}"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUT / (stem + "." + extension), dpi=320, facecolor="white")
    plt.close(figure)
    with fitz.open(OUT / (stem + ".pdf")) as document:
        page = document[0]
        outside = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if not page.rect.contains(fitz.Rect(span["bbox"])):
                        outside.append(dict(text=span["text"], bbox=span["bbox"]))
        assert not outside, (stem, outside)
    return dict(stem=stem, method=method, points=24, labels=labels,
                label_font_pt=12. if PPT else 9.5, figure_inches=list(FIGURE_INCHES),
                x_limits=list(XLIMITS), y_limits=list(YLIMITS), y_scale="log",
                bubble_area_scale_points2_per_primitive=SIZE_SCALE,
                label_overlap=False, label_bubble_overlap=False, label_legend_overlap=False,
                leader_label_overlap=False, bubbles_drawn_largest_first_at_equal_zorder=True,
                leaders_avoid_unrelated_bubbles=True,
                text_outside_pdf_page=outside)


def write_gallery(figures):
    sections = []
    for item in figures:
        stem = item["stem"]
        links = " | ".join(f'<a href="{stem}.{ext}">{ext.upper()}</a>' for ext in ("png", "pdf", "svg"))
        sections.append(f'<section><h2>{escape(item["method"])}</h2><a href="{stem}.png">'
                        f'<img src="{stem}.png" alt="Basis accuracy and AO count for {escape(item["method"])}"></a>'
                        f'<p>{links}</p></section>')
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Non-QZ basis accuracy</title><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.5 Arial;color:#171717}'
        'img{width:100%;height:auto}a{color:#0057b8}section{margin:35px 0}</style>'
        '<h1>Non-QZ basis accuracy: DF-MP2 and ωB97M-V</h1>'
        '<p>24 complete bases, 30 physical structures, 6 ISO34 isomerizations and 6 S22 dimer interactions. '
        'The ordinate is the equally weighted mean of the two category MAEs, in kcal/mol. '
        'Both plots share their axes and circle-area scale. Circle area encodes the mean number of '
        'unique spherical primitive GTOs across all 30 structures; it is not a runtime measurement. '
        'Red labels omit the NNAO prefix.</p>'
        '<p>Database-reference errors combine method and basis effects. This is a partial non-QZ campaign, '
        'with no counterpoise correction and no independent-holdout claim. ISO34 uses W1-F12 references; '
        'S22B uses estimated CCSD(T)/CBS references.</p>'
        '<p><a href="basis-accuracy-comparison.pdf">Two-page PDF</a> | '
        '<a href="basis-accuracy-pngs.zip">Both PNGs</a> | '
        '<a href="plotted-values.csv">Exact plotted values</a></p>' + ''.join(sections) + '</html>')
    if PPT:
        index = ROOT / "output/ppt/index.html"
        text = index.read_text()
        section = ('<section id="basis-accuracy-benchmark"><h2>Non-QZ basis accuracy</h2>'
                   '<p><a href="basis_accuracy_benchmark/index.html">DF-MP2 and ωB97M-V: '
                   '24 bases on matched reference differences</a></p></section>')
        if 'id="basis-accuracy-benchmark"' not in text:
            index.write_text(text.replace('<section', section + '<section', 1))


def write_readme(rows):
    nnao = sorted((r for r in rows if r["kind"] == "nnao" and r["method"] == "mp2"),
                  key=lambda r: ({"S": 0, "D": 1, "T": 2}[r["label"].split("-")[1][0]],
                                 {"S": 0, "M": 1, "L": 2}[r["label"].split("-")[-1]]))
    lineage = "\n".join(f'| {r["label"]} | {r["update_step"]} | {r["checkpoint_sha256"]} |' for r in nnao)
    (OUT / "README.md").write_text("""# Non-QZ basis accuracy benchmark

- [DF-MP2 PNG](basis-accuracy-mp2.png) / [PDF](basis-accuracy-mp2.pdf) / [SVG](basis-accuracy-mp2.svg)
- [omegaB97M-V PNG](basis-accuracy-wb97m-v.png) / [PDF](basis-accuracy-wb97m-v.pdf) / [SVG](basis-accuracy-wb97m-v.svg)
- [Two-page PDF](basis-accuracy-comparison.pdf) / [Both PNGs](basis-accuracy-pngs.zip)
- [Gallery](index.html) / [Matched comparison preview](comparison-overview.png)
- [Exact plotted source table](plotted-values.csv) / [Plotting record](plot-report.json)
- [Figure verification](verification.json) / [Independent saved-data audit](data-audit.json)

## Source and meaning

The unchanged Case 18 handoff is under
`data/computational_examples/2026-10-29/cases/18_basis_accuracy_partial_non_qz/`.
It contains 24 complete bases (15 conventional, nine NNAO), 30 physical
structures, 1,440 saved electronic-energy records, and 12 database reference
energy differences: six ISO34 isomerizations and six S22 dimer interactions.
All included bases use the identical structures and observables.

Each point is one basis with one electronic-structure method. The abscissa
is the arithmetic mean of contracted spherical AO counts across all 30
structures, including fragments. The ordinate is the equally weighted mean
of the isomerization MAE and interaction MAE, in kcal/mol. Each category
has six observations, so this equals the MAE of all 12 reference differences.
Both methods use the same linear AO range and logarithmic MAE range.

Circle area is proportional to the mean number of unique spherical primitive
GTO functions across the same 30 structures. Core functions and angular
degeneracy are included, and repeated contraction columns do not duplicate
a primitive. Every data point and size-legend example uses the same circular
marker and the same area mapping. Point coordinates are never jittered;
overlapping basis points remain at their actual values and keep separate
name labels. Larger circles are drawn first at the same layer as smaller
circles, so small points in dense clusters remain visible.

This is combined method-and-basis accuracy against database references.
It is not an absolute-total-energy accuracy score or a same-method basis-
limit error. The three QZ comparison bases and the aug-cc-pVQZ finite-basis
reference are outside this partial snapshot. Interactions use independently
evaluated physical fragments, E(AB)-E(A)-E(B), without CP or BSSE correction.
The methane fragments from S22/08 have a documented relation to the training
sources, so this set is not presented as an independent holdout.

The reference methods remain ISO34/W1-F12 and S22B/estimated CCSD(T)/CBS.
The word estimated is retained for S22B. Exact values, source signs, levels,
DOIs and upstream identities are preserved in the archive reference tables.
DF-MP2 uses the recorded frozen-core protocol; omegaB97M-V includes its
nonlocal correlation and uses main/NLC grid level 3. The mixed V100/A800
records are not used to construct a runtime ranking.

## Appearance and delivery

Both figures retain the original 8.8 x 7.6 inch proportions. The presentation
profile uses bold DejaVu Sans, 17-pt axis labels, 14-pt ticks, 12-pt direct
labels, 1.5-pt black frames, and inward ticks on all four sides. No grid is
drawn. Family and primitive-size legends are inside the frame. NNAO points
and shortened names are red; cc-pVTZ is emphasized in blue. Red labels omit
the NNAO prefix. Leader positions are recomputed for the larger labels,
without moving any measured point. Leaders avoid labels and unrelated
bubbles. PNGs are 2,816 x 2,432 pixels at 320 dpi; PDF and SVG preserve vector
text and markers. The overview places the two images beside each other
without changing either image's proportions.

## Frozen NNAO identities

These are the nine family-specific frozen models in this new handoff,
not the historical update-3351 model used for earlier illustrative figures.
Full source and parameter identities remain in the exported basis records.

| Basis | Update | Frozen checkpoint SHA256 |
|---|---:|---|
""" + lineage + """

## Reproduction and verification

```bash
/opt/anaconda3/bin/python scripts/plot_basis_accuracy_benchmark.py --ppt
```

The plotter reads the saved tables only and exports their exact 48 plotted
scores with unchanged AO and primitive coordinates. The independent audit
checks all 1,477 source-manifest hashes, the 1,440 archived statuses, the
720 explicit basis/structure inventories and all 576 relative energies.
Its maximum MAE reconstruction residual is approximately 1.78e-15 kcal/mol.
No SCF, model inference, training, optimization, remote job or code test
suite is run. Existing manuscript content and scientific archives are not
edited by the plotting workflow.
""")


def main():
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [SOURCE, CASE / "tables/relative-energies.csv", CASE / "tables/basis-costs.csv",
              CASE / "inputs/observables.json", CASE / "inputs/reference-levels.json",
              CASE / "snapshot-manifest.json", CASE / "verification.json"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    rows = read_rows()
    figures = [draw_plot(rows, method, title) for method, title in METHODS]
    shutil.copyfile(SOURCE, OUT / "plotted-values.csv")
    pdf = PdfWriter()
    for item in figures:
        pdf.append(PdfReader(OUT / (item["stem"] + ".pdf")))
    pdf.write(OUT / "basis-accuracy-comparison.pdf")
    with zipfile.ZipFile(OUT / "basis-accuracy-pngs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for item in figures:
            name = item["stem"] + ".png"
            archive.write(OUT / name, arcname=name)
    with Image.open(OUT / (figures[0]["stem"] + ".png")) as first, Image.open(OUT / (figures[1]["stem"] + ".png")) as second:
        preview = Image.new("RGB", (first.width * 2, first.height), "white")
        preview.paste(first, (0, 0)); preview.paste(second, (first.width, 0))
        preview.thumbnail((1760, 760), Image.Resampling.LANCZOS)
        preview.save(OUT / "comparison-overview.png")
    assert all(sha(ROOT / name) == digest for name, digest in hashes.items())
    record = dict(figures=figures, style=style_metadata(), plotted_points=48,
        cohort=dict(bases=24, conventional=15, nnao=9, structures=30, isomerizations=6, dimers=6),
        reference="Frozen ISO34 W1-F12 and S22B estimated CCSD(T)/CBS database energy differences",
        ordinate="0.5 * (isomerization MAE + dimer interaction MAE), equal to all-12-observable MAE",
        abscissa="Arithmetic mean of contracted spherical AO counts over all 30 structures",
        bubble_area="One common circle area proportional to mean unique spherical primitive GTO count",
        cp_bsse_included=False, source_sha256=hashes, script_sha256=sha(Path(__file__)),
        data_scope="Complete non-QZ snapshot, partial larger campaign; combined method/basis accuracy; not an independent holdout or runtime comparison.",
        scope="Saved-table plotting only; no model inference, SCF, optimization or training.")
    (OUT / "plot-report.json").write_text(json.dumps(record, indent=2) + "\n")
    (OUT / "verification.json").write_text(json.dumps(dict(
        figures=2, plotted_points=48, all_values_read_from_source_summary=True,
        same_axes_and_circle_area_scale=True, all_24_labels_per_plot=True,
        label_overlap=False, label_bubble_overlap=False, label_legend_overlap=False,
        leader_label_overlap=False, leaders_avoid_unrelated_bubbles=True,
        bubbles_drawn_largest_first_at_equal_zorder=True,
        all_pdf_text_within_page=True, all_source_inputs_unchanged=True,
        combined_pdf_pages=2, source_sha256=hashes), indent=2) + "\n")
    write_gallery(figures)
    write_readme(rows)
    print(json.dumps(dict(figures=2, points=48, output=str(OUT.relative_to(ROOT)))))


if __name__ == "__main__":
    main()
