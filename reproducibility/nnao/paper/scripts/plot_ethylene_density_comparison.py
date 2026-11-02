"""Replot ethylene density planes: methods in rows, density/difference in columns.

Use --ppt. Reads the saved float64 density arrays only; no SCF, model inference
or AO/density evaluation is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from plot_case08_reactions import ROOT, configure_style, sha
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
import numpy as np
from pypdf import PdfReader, PdfWriter


CASE = ROOT / "data/computational_examples/2026-10-27/cases/17_ethylene_density_planes"
SOURCE = CASE / "arrays/plane-densities.npz"
OUT = ROOT / "output/ppt/ethylene_density_comparison"
MANUSCRIPT_OUT = ROOT / "output/manuscript_density"
METHODS = [("mace", "MACE"), ("free", "Free contraction"), ("sto6g-p", "STO-6G+P")]
DENSITY_LIMITS = [0., 1.5]
DIFFERENCE_LIMIT = .1
CONTOURS = [.005, .02, .08, .32]
DIFFERENCE_CONTOURS = [-.02, 0., .02]
DENSITY_STYLE = {"font.weight": "bold", "axes.labelweight": "bold", "axes.labelsize": 14,
                 "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.5,
                 "xtick.major.size": 4, "ytick.major.size": 4,
                 "xtick.major.width": 1.2, "ytick.major.width": 1.2,
                 "mathtext.default": "bf"}


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def label_contours(ax, contours, kind, fontsize=8.5):
    hints = ([(2., -1.6), (-1.8, -1.2), (1.6, -.5), (.3, -.3)]
             if kind == "density" else [(0., -.4), (1.8, 1.4), (0., 1.4)])
    labels = []
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    blocked = [text.get_window_extent(renderer).padded(3) for text in ax.texts]
    font = FontProperties(family="DejaVu Sans", weight="bold", size=fontsize)
    for index, (level, hint) in enumerate(zip(contours.levels, hints)):
        segments = [segment for segment in contours.allsegs[index] if len(segment)]
        if not segments:
            continue
        points = np.concatenate(segments)
        width, height, _ = renderer.get_text_width_height_descent(f"{level:g}", font, False)
        candidates = np.argsort(np.sum((points - hint) ** 2, axis=1))
        position = None
        label_box = None
        force_callout = kind == "difference" and level > 0
        inline_candidates = [] if force_callout else candidates[::max(1, len(candidates) // 800)]
        for candidate in inline_candidates:
            point = points[candidate]
            px, py = ax.transData.transform(point)
            box = Bbox.from_extents(px - width / 2 - 3, py - height / 2 - 4,
                                    px + width / 2 + 3, py + height / 2 + 4)
            if (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                    and not any(box.overlaps(other) for other in blocked)):
                position, label_box = point, box
                break
        if position is None:
            target = points[candidates[0]]
            if force_callout:
                core = points[(points[:, 0] > -.95) & (points[:, 0] < -.45) & (abs(points[:, 1]) < .25)]
                if len(core):
                    target = core[np.argmin(core[:, 0])]
            locations = ((-2., .5), (-2., -.5)) if force_callout else (
                (0., 1.65), (1.5, 1.65), (-1.5, -1.6), (0., -1.5))
            for location in locations:
                px, py = ax.transData.transform(location)
                box = Bbox.from_extents(px - width / 2 - 3, py - height / 2 - 4,
                                        px + width / 2 + 3, py + height / 2 + 4)
                if not any(box.overlaps(other) for other in blocked):
                    text = ax.annotate(f"{level:g}", xy=target, xytext=location,
                        fontsize=fontsize, fontweight="bold", ha="center", va="center", zorder=7,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=.3),
                        arrowprops=dict(arrowstyle="->", color="#5b5b5b", lw=.8, mutation_scale=7))
                    labels.append(dict(value=float(level), position=list(location), text=text.get_text(),
                                       contour_position=target.tolist(), placement="callout"))
                    blocked.append(box)
                    break
            else:
                raise ValueError(f"No clear contour label position: {kind} {level}")
            continue
        before = len(contours.labelTexts)
        ax.clabel(contours, levels=contours.levels, manual=[position], inline=True,
                  fontsize=fontsize, fmt=lambda value: f"{value:g}", inline_spacing=2)
        for text in contours.labelTexts[before:]:
            text.set_fontweight("bold")
            text.set_rotation(0)
            text.set_bbox(dict(facecolor="white", edgecolor="none", alpha=.8, pad=.2))
            labels.append(dict(value=float(text.get_text()), position=list(text.get_position()), text=text.get_text()))
            blocked.append(label_box)
    return labels


def value_label(value):
    if value != 0 and abs(value) < .001:
        exponent = int(np.floor(np.log10(abs(value))))
        return rf"{value / 10 ** exponent:.3g}\times10^{{{exponent}}}"
    return f"{value:.4g}"


def nuclei(ax, geometry, plane):
    vertical_axis, omitted_axis = (1, 2) if plane == "xy" else (2, 1)
    marked = []
    for i, (symbol, center) in enumerate(zip(geometry["symbols"], geometry["coords_angstrom"])):
        if abs(center[omitted_axis]) > 1e-10:
            continue
        x, y = center[0], center[vertical_axis]
        ax.scatter([x], [y], s=13, color="#171717", edgecolors="white", linewidths=.6, zorder=5)
        ax.annotate(symbol, (x, y), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=10, color="#171717", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.35))
        marked.append(i)
    return marked


def draw_plane(data, geometry, plane, *, manuscript=False):
    x, vertical = data["x_angstrom"], data["vertical_angstrom"]
    dx, dv = x[1] - x[0], vertical[1] - vertical[0]
    extent = [x[0] - dx / 2, x[-1] + dx / 2,
              vertical[0] - dv / 2, vertical[-1] + dv / 2]
    reference = data[f"{plane}_cc-pvtz"]
    density_norm = Normalize(vmin=DENSITY_LIMITS[0], vmax=DENSITY_LIMITS[1])
    difference_norm = Normalize(vmin=-DIFFERENCE_LIMIT, vmax=DIFFERENCE_LIMIT)
    density_cmap = plt.colormaps["Blues"].with_extremes(under="white")
    figure_inches = (6.7, 8.5) if manuscript else (9., 7.)
    figure = plt.figure(figsize=figure_inches)
    if manuscript:
        # Physical map dimensions preserve x:y/z scaling while narrowing the gap.
        map_width, map_height = 2.6 / 6.7, 2.2 / 8.5
        lefts, bottoms = (.70 / 6.7, 3.55 / 6.7), tuple(y / 8.5 for y in (5.87, 3.51, 1.15))
        centers = [left + map_width / 2 for left in lefts]
        output_dir = MANUSCRIPT_OUT
    else:
        map_width, map_height = .26, .28
        lefts, bottoms, centers = (.10, .575), (.68, .375, .07), (.23, .705)
        output_dir = OUT
    for center, title in zip(centers, (r"$\rho$", r"$\rho-\rho_{\mathrm{cc-pVTZ}}$")):
        figure.text(center, .977 if manuscript else .99, title, ha="center", va="top", fontsize=14)
    if not manuscript:
        figure.text(.475, .99, r"$xy,\ z=0$" if plane == "xy" else r"$xz,\ y=0$",
                    ha="center", va="top", fontsize=12)
    panels = []
    for row, ((key, label), bottom) in enumerate(zip(METHODS, bottoms)):
        rho = data[f"{plane}_{key}"]
        delta = rho - reference
        assert rho.shape == delta.shape == (561, 641)
        assert np.all(np.isfinite(rho)) and np.all(rho >= 0) and np.all(np.isfinite(delta))
        for column, left in enumerate(lefts):
            ax = figure.add_axes([left, bottom, map_width, map_height])
            field = rho if column == 0 else delta
            image = ax.imshow(field, origin="lower", extent=extent, aspect="equal", interpolation="nearest",
                              cmap=density_cmap if column == 0 else "RdBu_r",
                              norm=density_norm if column == 0 else difference_norm)
            assert np.array_equal(np.asarray(image.get_array()), field)
            if column == 0:
                contours = ax.contour(x, vertical, rho, levels=CONTOURS,
                                      colors="#5b5b5b", linewidths=.7, alpha=.9)
                density_image = image
            else:
                contours = ax.contour(x, vertical, delta, levels=DIFFERENCE_CONTOURS,
                                      colors="#5b5b5b", linewidths=.7, alpha=.9,
                                      linestyles=["--", "--", ":", "-", "-"])
                difference_image = image
            marked = nuclei(ax, geometry, plane)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(vertical[0], vertical[-1])
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.tick_params(top=True, right=True, labelbottom=row == 2, labelleft=column == 0)
            if row == 2:
                ax.set_xlabel("x along C=C (Å)", labelpad=4)
            if column == 0:
                ax.set_ylabel(("y" if plane == "xy" else "z") + " (Å)", labelpad=4)
            tag = f"({chr(97 + 2 * row + column)})" + (" " + label if column == 0 else "")
            ax.text(.025, .97, tag, transform=ax.transAxes, fontsize=12,
                    ha="left", va="top", color="#171717", zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=1.))
            extrema = rf"$\min={value_label(float(field.min()))},\ \max={value_label(float(field.max()))}$"
            ax.text(.025, .025, extrema, transform=ax.transAxes, fontsize=9.5 if manuscript else 9,
                    ha="left", va="bottom", color="#171717", zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=.5))
            contour_labels = label_contours(ax, contours, "density" if column == 0 else "difference",
                                            fontsize=9.5 if manuscript else 8.5)
            panels.append(dict(row=row, column=column, method=key, method_label=label,
                field="total_density" if column == 0 else "density_minus_cc_pvtz",
                source_array_key=f"{plane}_{key}", reference_array_key=f"{plane}_cc-pvtz" if column else None,
                shape=list(field.shape), minimum=float(field.min()), maximum=float(field.max()),
                displayed_array_sha256=digest(field), nuclei_indices=marked,
                nuclei_count=len(marked), source_values_preserved=True,
                annotated_extrema="Unclipped source-field minimum and maximum",
                labeled_contours=contour_labels,
                saturated_pixels_below=int(np.sum(field < (0. if column == 0 else -DIFFERENCE_LIMIT))),
                saturated_pixels_above=int(np.sum(field > (DENSITY_LIMITS[1] if column == 0 else DIFFERENCE_LIMIT))),
                axes_bounds=list(ax.get_position().bounds)))
    density_axis = figure.add_axes([lefts[0], .52 / 8.5, map_width, .07 / 8.5]
                                   if manuscript else [.385, .11, .018, .80])
    bar = figure.colorbar(density_image, cax=density_axis, extend="max",
                         orientation="horizontal" if manuscript else "vertical")
    bar.set_ticks([0., .5, 1., 1.5] if manuscript else [0., .3, .6, .9, 1.2, 1.5])
    bar.ax.tick_params(labelsize=11, width=1., length=3)
    bar.outline.set_linewidth(1.2)
    bar.set_label(r"$\rho$ ($e\,a_0^{-3}$)", fontsize=12, labelpad=6)
    difference_axis = figure.add_axes([lefts[1], .52 / 8.5, map_width, .07 / 8.5]
                                     if manuscript else [.86, .11, .018, .80])
    bar = figure.colorbar(difference_image, cax=difference_axis, extend="both",
                         orientation="horizontal" if manuscript else "vertical")
    bar.set_ticks([-.1, 0., .1] if manuscript else [-.1, -.05, 0., .05, .1])
    bar.ax.tick_params(labelsize=11, width=1., length=3)
    bar.outline.set_linewidth(1.2)
    bar.set_label(r"$\Delta\rho$ ($e\,a_0^{-3}$)", fontsize=12, labelpad=6)
    stem = f"ethylene-density-{plane}-comparison"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output_dir / (stem + "." + extension), dpi=320)
    plt.close(figure)
    return dict(plane=plane, stem=stem, panels=panels, figure_inches=list(figure_inches),
                plane_location_in_image=not manuscript,
                colorbar_orientation="horizontal" if manuscript else "vertical")


def build_manuscript():
    """Render the compact manuscript layout without changing the PPT figures."""
    configure_style()
    MANUSCRIPT_OUT.mkdir(parents=True, exist_ok=True)
    geometry = json.loads((CASE / "inputs/structure.json").read_text())
    with np.load(SOURCE) as saved:
        data = {key: saved[key] for key in saved.files}
    with plt.rc_context(DENSITY_STYLE):
        figures = [draw_plane(data, geometry, plane, manuscript=True) for plane in ("xy", "xz")]
    reviewed = json.loads((OUT / "plot-report.json").read_text())
    original = {(f["plane"], p["row"], p["column"]): p
                for f in reviewed["figures"] for p in f["panels"]}
    for figure in figures:
        for panel in figure["panels"]:
            previous = original[figure["plane"], panel["row"], panel["column"]]
            assert panel["displayed_array_sha256"] == previous["displayed_array_sha256"]
            assert panel["minimum"] == previous["minimum"] and panel["maximum"] == previous["maximum"]
    report = dict(figures=figures, rows=[key for key, _ in METHODS],
        columns=["total density", "density minus cc-pVTZ"],
        figure_inches=[6.7, 8.5], map_inches=[2.6, 2.2], column_gap_inches=.25,
        plane_location="Caption only; no plane-position annotation inside the figures.",
        coordinate_aspect="Equal physical scales; full archived windows preserved.",
        color_scales="Linear; density [0,1.5], difference [-0.1,0.1] electron/bohr^3.",
        all_twelve_fields_equal_reviewed_grids=True,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                       (SOURCE, CASE / "inputs/structure.json", OUT / "plot-report.json", Path(__file__))})
    (MANUSCRIPT_OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads((CASE / "results.json").read_text())
    assert results["status"] == "complete" and results["passed"] == 6
    geometry = json.loads((CASE / "inputs/structure.json").read_text())
    with np.load(SOURCE) as saved:
        data = {key: saved[key] for key in saved.files}
    assert data["xy_z_angstrom"] == data["xz_y_angstrom"] == 0.
    with plt.rc_context(DENSITY_STYLE):
        figures = [draw_plane(data, geometry, plane) for plane in ("xy", "xz")]
    book = PdfWriter()
    cards = []
    for figure in figures:
        stem = figure["stem"]
        book.append(PdfReader(OUT / (stem + ".pdf")))
        cards.append('<section><h2>' + figure["plane"] + ' plane</h2><a href="' + stem + '.png">'
            '<img src="' + stem + '.png" alt="Ethylene ' + figure["plane"] + ' total and difference density"></a><p>' +
            ' | '.join('<a href="' + stem + '.' + ext + '">' + ext.upper() + '</a>'
                       for ext in ("png", "pdf", "svg")) + '</p></section>')
    book.write(OUT / "ethylene-density-comparisons.pdf")
    report = dict(figures=figures, rows=[key for key, _ in METHODS],
        columns=["total electron density", "density minus cc-pVTZ"],
        difference_definition="rho_method - rho_cc-pVTZ, computed from unmodified linear density arrays",
        density_units="electron / bohr^3", coordinate_units="Angstrom",
        density_color_scale=dict(kind="linear", limits=DENSITY_LIMITS, extend="max"),
        difference_color_scale=dict(kind="linear", limits=[-DIFFERENCE_LIMIT, DIFFERENCE_LIMIT], extend="both",
                                   positive="red", negative="blue", zero="white"),
        contour_levels=dict(density=CONTOURS, difference=DIFFERENCE_CONTOURS),
        color_saturation="Shared valence-focused display limits; extension triangles denote saturated core values. Raw arrays and annotated extrema are unchanged.",
        shared_color_scales_across_methods_and_planes=True, interpolation="nearest; no density smoothing",
        figure_inches=[9., 7.], png_dpi=320, font_weight="bold", axis_label_pt=14, tick_label_pt=12,
        source_sha256={str(p): sha(p) for p in [SOURCE, CASE / "inputs/structure.json", CASE / "results.json",
            CASE / "plot-metadata.json", CASE / "README.md", Path(__file__), ROOT / "scripts/plot_case08_reactions.py"]},
        scope="Two-dimensional total-electron-density slices, including core electrons; no projection integration. "
              "The xz plane marks only the two carbon nuclei lying in that plane. "
              "cc-pVTZ is a matched DF-RHF basis reference. No SCF or density reevaluation.")
    (OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Ethylene density comparison</title><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.5 Arial;color:#171717}'
        'img{width:100%;height:auto}a{color:#0057b8}section{margin:35px 0}</style>'
        '<h1>Ethylene total and difference density</h1><p>Rows: MACE, free contraction, STO-6G+P. '
        'Both columns use shared linear color scales. Left: total density, 0 to 1.5 electron/bohr^3. '
        'Right: method minus cc-pVTZ, -0.1 to +0.1 electron/bohr^3; red is positive and blue negative. '
        'Values outside the displayed ranges saturate, as indicated by colorbar extension triangles. '
        'Panel minima and maxima are the original full-field values. Contour numbers use the same density units. '
        'Coordinates are Angstrom.</p>' + ''.join(cards) + '</html>')
    print(json.dumps({key: value for key, value in report.items() if key not in ("figures", "source_sha256", "scope")}, indent=2))


if __name__ == "__main__":
    if "--manuscript" in sys.argv:
        build_manuscript()
    else:
        main()
