"""Plot two corrected dimer interactions side by side in one paper column.

Reads the archived Case 15 isolated-reference curves and existing molecular
renders only. No model inference or electronic-structure calculation is run.
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
import fitz
from PIL import Image, ImageChops

CASE = ROOT / "data/computational_examples/2026-10-26/cases/15_dimer_cp_output_controls"
SOURCE = CASE / "combined/curves.csv"
DEFINITIONS = CASE / "inputs/cases.json"
STRUCTURES = CASE / "representative-structures/index.json"
RENDERS = ROOT / "output/case08_xyzrender"
PREVIEWS = ROOT / "output/manuscript_layout/dimers"
PREVIOUS_VALUES = ROOT / "output/ppt/case15_dimer_controls/plotted-no-cp-values.csv"
SELECTED = [("Water-Water", "(a) Water dimer"),
            ("Benzene-Benzene_pi-pi", "(b) Benzene dimer")]
SERIES = [
    ("Reference", "Reference", "#171717", "-", .85, None),
    ("sto-6g", "STO-6G", "#89939c", "-", .65, "o"),
    ("STO-6G+P", "STO-6G+P", "#8A56A2", ":", .75, "D"),
    ("6-31g(d,p)", "6-31G(d,p)", "#8D933E", "-.", .75, "o"),
    ("def2-svp", "def2-SVP", "#ba718b", "--", .75, "o"),
    ("cc-pvdz", "cc-pVDZ", "#5C8C9E", ":", .75, "o"),
    ("cc-pvtz", "cc-pVTZ", "#0057B8", "-", 1.25, "o"),
    ("NNAO/MACE", "NNAO/MACE", "#ef233c", "--", 1.0, "s"),
]
FIGURE_INCHES = (3.4, 3.25)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def structure_picture(record):
    path = RENDERS / record["png"]
    assert sha(path) == record["png_sha256"]
    with Image.open(path) as original:
        rgba = original.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    if box is None:
        raise ValueError(f"Empty structure image: {path}")
    pad = 12
    box = (max(0, box[0] - pad), max(0, box[1] - pad),
           min(image.width, box[2] + pad), min(image.height, box[3] + pad))
    return image.crop(box), path


def build_figure(target: Path) -> dict:
    """Render the selected two dimer panels and return source/layout metadata."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    source_paths = [SOURCE, DEFINITIONS, STRUCTURES, PREVIOUS_VALUES]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    all_rows = read_csv(SOURCE)
    assert len(all_rows) == 1728
    rows = [row for row in all_rows if row["cp_mode"] == "no_cp"
            and row["basis"] != "NNAO/MACE-fixed"]
    assert len(rows) == 756
    definitions = json.loads(DEFINITIONS.read_text())["definitions"]
    renders = {row["id"]: row for row in json.loads(STRUCTURES.read_text())["records"]}
    previous = {(row["group"], row["basis"], float(row["scale"])): float(row["interaction_kcal_mol"])
                for row in read_csv(PREVIOUS_VALUES) if row["dataset"] == "S66x8"}
    style = {"font.family": "DejaVu Sans", "font.weight": "bold", "axes.labelweight": "bold",
             "axes.titleweight": "bold",
             "mathtext.fontset": "dejavusans", "mathtext.default": "bf", "font.size": 7.,
             "axes.labelsize": 7., "xtick.labelsize": 6., "ytick.labelsize": 6.,
             "axes.linewidth": .7, "xtick.direction": "in", "ytick.direction": "in",
             "xtick.major.width": .55, "ytick.major.width": .55,
             "xtick.major.size": 2.3, "ytick.major.size": 2.3,
             "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"}
    figures, exported = [], []
    with plt.rc_context(style):
        figure = plt.figure(figsize=FIGURE_INCHES)
        width, height = FIGURE_INCHES
        plot_width = 1.25
        for col, ((group, title), left) in enumerate(zip(SELECTED, (.44, 2.0))):
            points = sorted((row for row in definitions if row["dataset"] == "S66x8"
                             and row["group"] == group), key=lambda row: row["scale"])
            assert len(points) == 8
            x = [point["scale"] for point in points]
            selected_rows = [row for row in rows if row["dataset"] == "S66x8" and row["group"] == group]
            lookup = {(row["basis"], float(row["scale"])): row for row in selected_rows}
            anchor = next(point for point in points if point["scale"] == 1.)
            record = renders[anchor["identifier"]]
            picture, picture_path = structure_picture(record)
            source_hashes[str(picture_path.relative_to(ROOT))] = sha(picture_path)
            ax = figure.add_axes([left / width, .35 / height, plot_width / width, 2.65 / height])
            ax.set_title(title, fontsize=6.7, loc="left", pad=5.)
            values, handles = [], []
            for basis, label, color, linestyle, linewidth, marker in SERIES:
                if basis == "Reference":
                    energies = [point["reference_kcal_mol"] for point in points]
                else:
                    energies = [float(lookup[basis, scale]["interaction_kcal_mol"]) for scale in x]
                    assert len({(lookup[basis, scale]["isolated_a_hartree"],
                                 lookup[basis, scale]["isolated_b_hartree"]) for scale in x}) == 1
                assert all(energy == previous[group, basis, scale] for scale, energy in zip(x, energies))
                marker_size = 1.9 if basis in ("cc-pvtz", "NNAO/MACE") else 1.45
                ax.plot(x, energies, color=color, ls=linestyle, lw=linewidth,
                        marker=marker, ms=marker_size, markeredgewidth=.35, zorder=3)
                handles.append(Line2D([], [], color=color, ls=linestyle, lw=linewidth,
                                      marker=marker, ms=marker_size, label=label))
                values.extend(energies)
                exported.extend(dict(dataset="S66x8", group=group, cp_mode="no_cp", basis=basis,
                                     scale=scale, interaction_kcal_mol=float(energy))
                                for scale, energy in zip(x, energies))
            low, high = min(min(values), 0.), max(max(values), 0.)
            span = high - low
            lower, upper = low - .10 * span, high + .55 * span
            ax.set(xlim=(.85, 2.035), xticks=[1., 1.5, 2.])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlabel("Distance scale", labelpad=3.)
            ax.tick_params(top=True, right=True, pad=2.)
            if col == 0:
                ax.set_ylabel(r"Interaction energy (kcal mol$^{-1}$)", labelpad=3.)
            ax.axhline(0., color=".5", lw=.4, zorder=0)
            legend = ax.legend(handles=handles, loc="upper left", fontsize=5.5,
                               handlelength=1.25, handletextpad=.35,
                               labelspacing=.18, borderaxespad=.32)
            inset_width = .34
            inset_height = inset_width * (plot_width / 2.65) * picture.height / picture.width
            if inset_height > .23:
                inset_height = .23
                inset_width = inset_height * (2.65 / plot_width) * picture.width / picture.height
            inset_box = [.98 - inset_width, .98 - inset_height, inset_width, inset_height]
            inset = ax.inset_axes(inset_box)
            inset.imshow(picture)
            inset.set_axis_off()
            for _ in range(20):
                ax.set_ylim(lower, upper)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                legend_box = legend.get_window_extent(renderer)
                image_box = inset.get_window_extent(renderer)
                assert not legend_box.overlaps(image_box), (group, "legend-image overlap")
                legend_low = ax.transData.inverted().transform((0., legend_box.y0))[1]
                image_low = lower + inset_box[1] * (upper - lower)
                if min(legend_low, image_low) > high:
                    break
                upper += .1 * span
            assert min(legend_low, image_low) > high
            figures.append(dict(dataset="S66x8", group=group, geometry_points=8, curves=8,
                                y_limits=[lower, upper], source_structure=record["id"],
                                inset_axes_box=inset_box, legend_bottom_energy=float(legend_low),
                                all_data_below_legend_and_inset=True, legend_image_overlap=False))
        figure.savefig(target, dpi=600, facecolor="white")
        png = PREVIEWS / (target.stem + ".png")
        figure.savefig(png, dpi=400, facecolor="white")
        plt.close(figure)
    assert len(exported) == 128
    with (PREVIEWS / "plotted-values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    with fitz.open(target) as pdf:
        assert len(pdf) == 1
        page = pdf[0]
        outside = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for text in line["spans"]:
                    if not page.rect.contains(fitz.Rect(text["bbox"])):
                        outside.append(dict(text=text["text"], bbox=text["bbox"]))
        assert not outside, outside
        page_points = list(page.rect)
    assert all(sha(ROOT / path) == value for path, value in source_hashes.items())
    report = dict(target=str(target.relative_to(ROOT)), target_sha256=sha(target),
                  figure_inches=list(FIGURE_INCHES), page_points=page_points,
                  layout="Two side-by-side dimer panels in a single manuscript column",
                  panels=figures, curves_per_panel=8, plotted_values=128,
                  source_values_match_reviewed_plots=True, points_per_curve=8,
                  cp_mode="no_cp", mace_policy="Adaptive only; fixed network weights and exponents",
                  excluded_basis="NNAO/MACE-fixed", legend_font_pt=5.5,
                  axis_label_pt=7., tick_label_pt=6., source_sha256=source_hashes,
                  source_artifacts_unchanged=True, script_sha256=sha(Path(__file__)),
                  text_outside_pdf_page=outside, preview=str(png.relative_to(ROOT)),
                  energy_definition="E_AB(R)-E_A(isolated)-E_B(isolated), with constant monomer references",
                  scope="Archived corrected no-CP interactions only; no SCF, model inference, fit, clipping, or endpoint shifting.")
    (PREVIEWS / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    record = build_figure(ROOT / "manuscript/figures/fig-dimer-interactions.pdf")
    print(json.dumps(dict(target=record["target"], plotted_values=record["plotted_values"],
                          figure_inches=record["figure_inches"])))


if __name__ == "__main__":
    main()
