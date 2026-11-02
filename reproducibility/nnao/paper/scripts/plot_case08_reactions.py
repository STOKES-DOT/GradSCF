"""Draw one structure-annotated energy diagram per BH9 reaction.

Uses archived energies and xyzrender SVGs; does not run electronic calculations.
Style follows the rendered GradTDDFT manuscript energy plots, with larger
typography than the earlier external redraw script.
Pass --ppt for larger, bold typography and thicker lines in output/ppt/.
"""
from __future__ import annotations

import csv
import hashlib
from html import escape
from io import BytesIO
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PPT = "--ppt" in sys.argv[1:]
OUTPUT_ROOT = ROOT / ("output/ppt" if PPT else "output")
LINE_SCALE = 1.7 if PPT else 1.
MARKER_SCALE = 1.4 if PPT else 1.
OUT = OUTPUT_ROOT / "case08_reaction_profiles"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from xyzrender.export import svg_to_png_bytes

CASE = ROOT / "data/computational_examples/2026-10-21/cases/08_distance_curves_and_barriers"
DATASET = CASE / "inputs/dataset.json"
RESULTS = CASE / "artifacts/nnao-curves-barriers-repair-20261021/merged/results.json"
FLAT = RESULTS.parent / "reaction-cases.csv"
RENDERS = ROOT / "output/case08_xyzrender"
REFERENCE_PDF = Path("/Users/jiaoyuan/Documents/GitHub/GradTDDFT_paper/figures/h2_ground_hfx_nopt2_dissociation_paper_style.pdf")
INSET_REFERENCE = Path("/Users/jiaoyuan/Documents/GitHub/GradTDDFT_paper/figures/qm9gwbse_two_system_selected_functional_spectra_with_structures_nnxc_reference_only.pdf")
BLACK, RED = "#171717", "#ef233c"
SERIES = [
    ("Reference", "BH9 reference", BLACK, "-", 1.7),
    ("6-31g(d,p)", "6-31G(d,p)", "#8D933E", "-.", 1.7),
    ("cc-pvdz", "cc-pVDZ", "#5C8C9E", ":", 1.7),
    ("cc-pvtz", "cc-pVTZ", "#0057B8", "-", 2.8),
    ("NNAO/MACE", "NNAO/MACE", RED, "--", 1.8),
]
SERIES = [(basis, label, color, linestyle, width * LINE_SCALE)
          for basis, label, color, linestyle, width in SERIES]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "font.weight": "bold" if PPT else "normal",
        "axes.labelweight": "bold" if PPT else "normal",
        "axes.labelsize": 10.75, "axes.titlesize": 10.5,
        "axes.linewidth": .9, "axes.edgecolor": BLACK,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": .8, "ytick.major.width": .8,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9,
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "svg.fonttype": "none", "mathtext.fontset": "dejavusans",
    })
    if PPT:
        plt.rcParams.update({
            "font.size": 14, "axes.labelsize": 17, "axes.titlesize": 17,
            "axes.linewidth": 1.5,
            "xtick.labelsize": 14, "ytick.labelsize": 14,
            "xtick.major.size": 5, "ytick.major.size": 5,
            "xtick.major.width": 1.3, "ytick.major.width": 1.3,
            "legend.fontsize": 12.5, "mathtext.default": "bf",
        })


def style_metadata() -> dict:
    return dict(profile="ppt" if PPT else "paper", font="DejaVu Sans",
                font_weight=plt.rcParams["font.weight"],
                axis_label_pt=plt.rcParams["axes.labelsize"],
                tick_label_pt=plt.rcParams["xtick.labelsize"],
                legend_pt=plt.rcParams["legend.fontsize"],
                border_width_pt=plt.rcParams["axes.linewidth"],
                line_width_multiplier=LINE_SCALE, marker_size_multiplier=MARKER_SCALE)


def high_resolution_structure(record: dict) -> Image.Image:
    """Rasterize the original vector at 3x, without changing its display camera."""
    svg = (RENDERS / record["svg"]).read_text()
    png = svg_to_png_bytes(svg, size=2700)
    image = Image.open(BytesIO(png)).convert("RGBA")
    white = Image.new("RGBA", image.size, "white")
    image = Image.alpha_composite(white, image).convert("RGB")
    foreground = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    if foreground is None:
        raise ValueError(f"Empty structure: {record['id']}")
    pad = 24
    box = (max(0, foreground[0] - pad), max(0, foreground[1] - pad),
           min(image.width, foreground[2] + pad), min(image.height, foreground[3] + pad))
    return image.crop(box)


def stage_images(definition: dict, renders: dict):
    stages = [list(definition["reactants"]), [definition["transition"]], list(definition["products"])]
    images = [[high_resolution_structure(renders["20_bh9/" + name]) for name in stage] for stage in stages]
    width, height, gap = 1500, 950, 110
    factor = min(min((width - gap * (len(row) - 1)) / sum(im.width for im in row),
                     (height - 60) / max(im.height for im in row)) for row in images)
    plus_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 100)
    output = []
    for names, row in zip(stages, images):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        sizes = [(max(1, round(im.width * factor)), max(1, round(im.height * factor))) for im in row]
        x = (width - sum(s[0] for s in sizes) - gap * (len(row) - 1)) // 2
        for number, (im, size) in enumerate(zip(row, sizes)):
            resized = im.resize(size, Image.Resampling.LANCZOS)
            canvas.paste(resized, (x, (height - size[1]) // 2))
            x += size[0]
            if number < len(row) - 1:
                draw.text((x + gap / 2, height / 2), "+", fill=BLACK, font=plus_font, anchor="mm")
                x += gap
        # This is a page composition of separate species, not a combined geometry.
        output.append((canvas, names))
    return output


def draw_reaction(definition, by_key, renders):
    rid = definition["identifier"]
    figure = plt.figure(figsize=(6.3, 6.3))
    ax = figure.add_axes([.15, .11, .81, .81])
    stage_x = [0., 1.4, 2.8]
    x_min, x_max = -.72, 3.52
    plateau_halfwidth, image_width = .61, 1.18
    ax.set_xlim(x_min, x_max)
    handles, levels_export = [], []
    values = []
    stage_levels = [[], [], []]
    for basis, label, color, linestyle, linewidth in SERIES:
        if basis == "Reference":
            energy = [0., definition["reference_forward_kcal_mol"], definition["reference_reaction_kcal_mol"]]
            if PPT:
                label = "Reference"
        else:
            energy = by_key[(rid, basis)]["relative_kcal_mol"]
        values.extend(energy)
        for position, level in enumerate(energy):
            x = stage_x[position]
            stage_levels[position].append(level)
            ax.plot([x - plateau_halfwidth, x + plateau_halfwidth], [level, level],
                    color=color, lw=linewidth, ls=linestyle, solid_capstyle="butt", zorder=3)
            levels_export.append(dict(reaction=rid, basis=basis, stage=("R", "TS", "P")[position],
                                      energy_kcal_mol=level))
        for i in range(2):
            ax.plot([stage_x[i] + plateau_halfwidth, stage_x[i + 1] - plateau_halfwidth],
                    energy[i:i + 2], color=color,
                    lw=linewidth, ls=linestyle, zorder=2)
        handles.append(Line2D([], [], color=color, lw=linewidth, ls=linestyle, label=label))
    span = max(values) - min(values)
    lower, upper = min(values) - .10 * span, max(values) + .45 * span
    ax.set_ylim(lower, upper)
    ax.axhline(0., color="0.45", lw=.55 * LINE_SCALE, zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xticks(stage_x, ["Reactants", r"TS$^{\ddagger}$", "Products"])
    ax.set_ylabel(r"Relative energy (kcal mol$^{-1}$)")
    ax.tick_params(top=True, right=True)
    downhill = definition["reference_forward_kcal_mol"] < 0
    legend_location = "lower left" if downhill else "upper left"
    legend = ax.legend(handles=handles, loc=legend_location, ncol=1,
                       handlelength=1.5 if PPT else 2., labelspacing=.3,
                       borderaxespad=.4 if PPT else .65)
    insets, anchors = [], []
    for position, (picture, names) in enumerate(stage_images(definition, renders)):
        width_fraction = image_width / (x_max - x_min)
        height_fraction = width_fraction * picture.height / picture.width
        image_bottom = max(stage_levels[position]) + .025 * span
        box = [(stage_x[position] - image_width / 2 - x_min) / (x_max - x_min),
               (image_bottom - lower) / (upper - lower), width_fraction, height_fraction]
        inset = ax.inset_axes(box)
        inset.imshow(picture)
        inset.set_axis_off()
        insets.append(inset)
        anchors.append(dict(stage=("R", "TS", "P")[position], species=names,
                            plateau_max_kcal_mol=max(stage_levels[position]),
                            image_bottom_kcal_mol=image_bottom, axes_box=box))
        picture.save(OUT / "structure-panels" / f"{rid}-{('R', 'TS', 'P')[position]}.png")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    assert all(not legend_box.overlaps(inset.get_window_extent(renderer)) for inset in insets), (
        rid, legend_box.bounds, [inset.get_window_extent(renderer).bounds for inset in insets])
    assert all(0 <= b["axes_box"][1] and sum(b["axes_box"][1::2]) <= 1 for b in anchors), rid
    stem = OUT / rid
    for extension in ("png", "pdf", "svg"):
        figure.savefig(str(stem) + "." + extension, dpi=320)
    plt.close(figure)
    return levels_export, dict(reaction=rid, legend_location=legend_location,
                               structure_anchors=anchors)


def write_index(definitions):
    cards = []
    for definition in definitions:
        rid = definition["identifier"]
        cards.append(f'<section id="{rid}"><h2>{escape(rid)}: {escape(definition["category_name"])}</h2>'
                     f'<a href="{rid}.pdf"><img src="{rid}.png" alt="{escape(rid)} reaction energy diagram with molecular structures"></a>'
                     f'<p><a href="{rid}.pdf">PDF</a> | <a href="{rid}.svg">SVG</a> | <a href="{rid}.png">PNG</a></p></section>')
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>BH9 reaction profiles</title>'
        '<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.5 Arial,sans-serif;color:#171717}'
        'img{width:100%;height:auto}section{margin:36px 0;border-top:1px solid #ddd}a{color:#175b9a}</style>'
        '<h1>BH9 reaction profiles with molecular structures</h1>'
        '<p>Nine individual reactions; archived DF-HF data and published BH9 reference values. '
        'Connectors are schematic, not IRC paths. Each method uses its own reactant zero. '
        'Molecular panels reuse the archived xyzrender views; separate species remain separate.</p>'
        '<p><a href="BH9-reactions-with-structures.pdf">All nine reactions (PDF)</a> | '
        '<a href="overview.png">Overview</a> | <a href="plotted-levels.csv">Plotted energy levels</a></p>'
        + ''.join(cards) + '</html>')


def main():
    configure_style()
    (OUT / "structure-panels").mkdir(exist_ok=True)
    dataset = json.loads(DATASET.read_text())
    results = json.loads(RESULTS.read_text())
    manifest = json.loads((RENDERS / "manifest.json").read_text())
    by_key = {(r["identifier"], r["basis"]): r for r in results["reactions"]}
    renders = {r["id"]: r for r in manifest["records"] if r["dataset"] == "BH9"}
    assert len(by_key) == 54 and len(renders) == 34
    for record in renders.values():
        assert sha(RENDERS / record["svg"]) == record["svg_sha256"]
    for row in csv.DictReader(FLAT.open()):
        actual = by_key[(row["identifier"], row["basis"])]
        assert actual["status"] == "passed"
        for key in ("forward_kcal_mol", "reverse_kcal_mol", "reaction_kcal_mol"):
            assert actual[key] == float(row[key])
    exported, layouts = [], []
    for definition in dataset["reactions"]:
        levels, layout = draw_reaction(definition, by_key, renders)
        exported.extend(levels)
        layouts.append(layout)
        print("Rendered", definition["identifier"], flush=True)
    with (OUT / "plotted-levels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader(); writer.writerows(exported)
    combined = PdfWriter()
    for definition in dataset["reactions"]:
        combined.append(PdfReader(OUT / (definition["identifier"] + ".pdf")))
    combined.write(OUT / "BH9-reactions-with-structures.pdf")
    sheet = Image.new("RGB", (2400, 2400), "white")
    for number, definition in enumerate(dataset["reactions"]):
        with Image.open(OUT / (definition["identifier"] + ".png")) as image:
            image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            sheet.paste(image, ((number % 3) * 800, (number // 3) * 800))
    sheet.save(OUT / "overview.png")
    write_index(dataset["reactions"])
    report = dict(figures=9, combined_pdf_pages=9, source_reaction_profiles=54,
        plotted_profiles=45, plotted_levels=len(exported), molecular_species=34,
        plotted_methods=[x[0] for x in SERIES], source_sha256={str(p): sha(p) for p in
            [DATASET, RESULTS, FLAT, RENDERS / "manifest.json", REFERENCE_PDF, INSET_REFERENCE, Path(__file__)]},
        style=dict(**style_metadata(), reference_width_pt=1.7 * LINE_SCALE,
                   model_width_pt=1.8 * LINE_SCALE, model_color=RED, reference_color=BLACK,
                   triple_zeta_color="#0057B8", triple_zeta_width_pt=2.8 * LINE_SCALE,
                   ticks="inward, four sides", grid=False, legend_frame=False,
                   figure_inches=[6.3, 6.3], png_dpi=320, legend_inside=True,
                   structure_placement="Directly above each stage's highest plotted platform"),
        layouts=layouts,
        scope="Archived electronic energies only; each method has its own R zero. "
            "Schematic R/TS/P connectors, not IRC. Separate species are combined in page layout only. "
            "Original xyzrender SVG views are rasterized at 2700 pixels for the molecular insets.")
    (OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_sha256", "scope", "layouts")}, indent=2))


if __name__ == "__main__":
    main()
