"""Redraw the two selected BH9 profiles at the manuscript's single-column size.

Archived energy levels and existing xyzrender structure panels are reused.
This script performs no model inference, SCF, optimization, or data fitting.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
import fitz


CASE = ROOT / "data/computational_examples/2026-10-21/cases/08_distance_curves_and_barriers"
DATASET = CASE / "inputs/dataset.json"
RESULTS = CASE / "artifacts/nnao-curves-barriers-repair-20261021/merged/results.json"
SOURCE = ROOT / "output/ppt/case08_reaction_profiles/plotted-levels.csv"
STRUCTURES = ROOT / "output/ppt/case08_reaction_profiles/structure-panels"
OUT = ROOT / "output/manuscript_layout/reactions"
IDS = ["BH9_02_134", "BH9_04_1"]
STAGES = ("R", "TS", "P")
SERIES = [
    ("Reference", "Reference", "#171717", "-", .95),
    ("6-31g(d,p)", "6-31G(d,p)", "#8D933E", "-.", .85),
    ("cc-pvdz", "cc-pVDZ", "#5C8C9E", ":", .95),
    ("cc-pvtz", "cc-pVTZ", "#0057B8", "-", 1.40),
    ("NNAO/MACE", "MACE", "#ef233c", "--", 1.05),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_figure(target: Path) -> dict:
    """Write one 3.4-inch-wide PDF containing two side-by-side reaction panels."""
    OUT.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    definitions = {r["identifier"]: r for r in json.loads(DATASET.read_text())["reactions"]}
    results = {(r["identifier"], r["basis"]): r
               for r in json.loads(RESULTS.read_text())["reactions"]}
    with SOURCE.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["reaction"] in IDS]
    by_key = {(r["reaction"], r["basis"], r["stage"]): float(r["energy_kcal_mol"])
              for r in rows}
    assert len(rows) == len(by_key) == 30
    source_paths = [DATASET, RESULTS, SOURCE]
    source_paths.extend(STRUCTURES / f"{rid}-{stage}.png" for rid in IDS for stage in STAGES)
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    figure_inches = (3.4, 3.1)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 6., "font.weight": "bold",
        "axes.labelweight": "bold", "axes.labelsize": 7., "axes.linewidth": .65,
        "axes.edgecolor": "#171717", "axes.facecolor": "white",
        "figure.facecolor": "white", "xtick.direction": "in", "ytick.direction": "in",
        "xtick.labelsize": 6., "ytick.labelsize": 6.,
        "xtick.major.size": 2., "ytick.major.size": 2.,
        "xtick.major.width": .55, "ytick.major.width": .55,
        "legend.frameon": False, "legend.fontsize": 5.6,
        "mathtext.fontset": "dejavusans", "mathtext.default": "bf",
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }):
        figure = plt.figure(figsize=figure_inches)
        axes = [figure.add_axes([left, .115, .365, .81]) for left in (.15, .62)]
        figure.text(.025, .52, r"Relative energy (kcal mol$^{-1}$)",
                    rotation=90, fontsize=7., ha="center", va="center")
        layouts, exported = [], []
        for panel, (rid, ax) in enumerate(zip(IDS, axes)):
            definition = definitions[rid]
            stage_levels = [[], [], []]
            curves = []
            handles = []
            for basis, label, color, linestyle, linewidth in SERIES:
                levels = [by_key[rid, basis, stage] for stage in STAGES]
                expected = ([0., definition["reference_forward_kcal_mol"],
                             definition["reference_reaction_kcal_mol"]] if basis == "Reference"
                            else results[rid, basis]["relative_kcal_mol"])
                assert np.array_equal(levels, expected), (rid, basis)
                curves.append(levels)
                for i, value in enumerate(levels):
                    stage_levels[i].append(value)
                    ax.plot([i - .435, i + .435], [value, value],
                            color=color, ls=linestyle, lw=linewidth,
                            solid_capstyle="butt", zorder=3)
                    exported.append(dict(reaction=rid, basis=basis, stage=STAGES[i],
                                         energy_kcal_mol=value))
                for i in range(2):
                    ax.plot([i + .435, i + 1 - .435], levels[i:i + 2],
                            color=color, ls=linestyle, lw=linewidth, zorder=2)
                handles.append(Line2D([], [], color=color, ls=linestyle,
                                      lw=linewidth, label=label))
            values = np.asarray(curves)
            span = float(values.max() - values.min())
            lower = float(values.min() - .10 * span)
            upper = float(values.max() + .65 * span)
            ax.set_xlim(-.5, 2.5)
            ax.set_ylim(lower, upper)
            ax.axhline(0., color=".6", lw=.4, zorder=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xticks([0., 1., 2.], ["R", "TS", "P"])
            ax.tick_params(top=True, right=True, pad=2.)
            ax.text(0., 1.025, "(a)" if panel == 0 else "(b)",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7.)
            legend = ax.legend(handles=handles, loc="upper left", handlelength=1.25,
                               handletextpad=.4, borderaxespad=.4, labelspacing=.25,
                               columnspacing=.5)
            image_boxes = []
            for i, stage in enumerate(STAGES):
                picture_path = STRUCTURES / f"{rid}-{stage}.png"
                with Image.open(picture_path) as source:
                    picture = source.convert("RGB")
                width_fraction = .84 / 3.
                # Inset height accounts for the tall, narrow physical plot frame.
                axes_inches = (figure_inches[0] * .365, figure_inches[1] * .81)
                height_fraction = width_fraction * axes_inches[0] / axes_inches[1] * picture.height / picture.width
                image_bottom = max(stage_levels[i]) + .025 * span
                box = [(i - .42 + .5) / 3., (image_bottom - lower) / (upper - lower),
                       width_fraction, height_fraction]
                inset = ax.inset_axes(box)
                inset.imshow(picture)
                inset.set_axis_off()
                image_boxes.append((inset, dict(stage=stage, path=str(picture_path.relative_to(ROOT)),
                    axes_box=box, bottom_kcal_mol=image_bottom,
                    highest_level_kcal_mol=max(stage_levels[i]))))
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            legend_box = legend.get_window_extent(renderer)
            assert ax.get_window_extent(renderer).contains(*legend_box.get_points()[0])
            assert ax.get_window_extent(renderer).contains(*legend_box.get_points()[1])
            assert all(not legend_box.overlaps(inset.get_window_extent(renderer))
                       for inset, record in image_boxes), rid
            assert all(0. <= record["axes_box"][1] and
                       record["axes_box"][1] + record["axes_box"][3] < 1.
                       for inset, record in image_boxes), rid
            # The lowest edge of the legend stays above every energy level.
            legend_bottom_data = ax.transData.inverted().transform([0., legend_box.y0])[1]
            assert legend_bottom_data > values.max(), (rid, legend_bottom_data)
            layouts.append(dict(reaction=rid, axes_box=list(ax.get_position().bounds),
                                y_limits=[lower, upper], legend_inside=True,
                                legend_bottom_kcal_mol=float(legend_bottom_data),
                                structures_above_stage_platforms=True,
                                structure_panels=[record for inset, record in image_boxes]))
        figure.savefig(target, dpi=600)
        figure.savefig(OUT / "fig-reaction-profiles.png", dpi=600)
        figure.savefig(OUT / "fig-reaction-profiles.svg", dpi=600)
        plt.close(figure)
    with (OUT / "plotted-levels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    with fitz.open(target) as doc:
        assert len(doc) == 1
        page = doc[0]
        spans = [s for b in page.get_text("dict")["blocks"]
                 for l in b.get("lines", []) for s in l["spans"]]
        outside = [dict(text=s["text"], bbox=s["bbox"]) for s in spans
                   if not page.rect.contains(fitz.Rect(s["bbox"]))]
        assert not outside, outside
        pdf_points = list(page.rect)
        fonts = sorted({s["font"] for s in spans})
    assert all(sha(ROOT / p) == digest for p, digest in input_hashes.items())
    report = dict(
        output=str(target.relative_to(ROOT)), output_sha256=sha(target),
        figure_inches=list(figure_inches), page_points=pdf_points,
        layout="one row, two panels, single column", selected_reactions=IDS,
        methods=[s[0] for s in SERIES], plotted_energy_levels=len(exported),
        all_30_energy_values_equal_archived_values=True,
        legend_label="MACE denotes the NNAO/MACE curve", legend_font_pt=5.6,
        tick_font_pt=6., axis_label_pt=7., panel_label_pt=7.,
        fonts=fonts, text_outside_page=outside, layouts=layouts,
        source_sha256=input_hashes, script_sha256=sha(Path(__file__)),
        source_artifacts_unchanged=True,
        scope="Archived electronic energy plotting only; no SCF or model inference.")
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    record = build_figure(ROOT / "manuscript/figures/fig-reaction-profiles.pdf")
    print(json.dumps({k: record[k] for k in ("output", "figure_inches", "layout", "plotted_energy_levels")}, indent=2))
