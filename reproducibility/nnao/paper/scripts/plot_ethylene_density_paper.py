"""Render the archived ethylene densities at a readable single-column size.

Only Case 17 saved arrays are read. No SCF, model inference, density
evaluation, smoothing, or numerical optimization is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
import numpy as np
import fitz


CASE = ROOT / "data/computational_examples/2026-10-27/cases/17_ethylene_density_planes"
SOURCE = CASE / "arrays/plane-densities.npz"
GEOMETRY = CASE / "inputs/structure.json"
PREVIEWS = ROOT / "output/manuscript_layout/density"
METHODS = [("mace", "MACE"), ("free", "Free contraction"), ("sto6g-p", "STO-6G+P")]
CONTOURS = [.005, .02, .08, .32]
DIFFERENCE_CONTOURS = [-.02, 0., .02]
FIGURE_INCHES = (3.4, 5.85)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(field):
    return hashlib.sha256(np.ascontiguousarray(field, dtype=np.float64).tobytes()).hexdigest()


def value_label(value):
    if value and abs(value) < .001:
        exponent = int(np.floor(np.log10(abs(value))))
        return rf"{value / 10 ** exponent:.3g}\times10^{{{exponent}}}"
    return f"{value:.4g}"


def nuclei(ax, geometry, plane):
    vertical, omitted = (1, 2) if plane == "xy" else (2, 1)
    marked = []
    for i, (symbol, xyz) in enumerate(zip(geometry["symbols"], geometry["coords_angstrom"])):
        if abs(xyz[omitted]) > 1e-10:
            continue
        ax.scatter([xyz[0]], [xyz[vertical]], s=5.5, color="#171717",
                   edgecolors="white", linewidths=.35, zorder=5)
        ax.annotate(symbol, (xyz[0], xyz[vertical]), xytext=(0, 3.7),
                    textcoords="offset points", ha="center", fontsize=6.2,
                    color="#171717", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=.15))
        marked.append(i)
    return marked


def contour_labels(ax, contours, kind):
    """Label each level once, avoiding nuclei, panel tags, and prior labels."""
    hints = ([(2., -1.6), (-1.8, -1.2), (1.6, -.5), (.3, -.3)]
             if kind == "density" else [(0., -.4), (1.8, 1.4), (-2., .5)])
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    blocked = [text.get_window_extent(renderer).padded(1.5) for text in ax.texts]
    font = FontProperties(family="DejaVu Sans", weight="bold", size=6.)
    labels = []
    for index, (level, hint) in enumerate(zip(contours.levels, hints)):
        segments = [seg for seg in contours.allsegs[index] if len(seg)]
        if not segments:
            raise ValueError(f"Missing requested contour {kind} {level}")
        points = np.concatenate(segments)
        width, height, _ = renderer.get_text_width_height_descent(f"{level:g}", font, False)
        candidates = np.argsort(np.sum((points - hint) ** 2, axis=1))
        position = box = None
        force_callout = kind == "difference" and level > 0
        for candidate in ([] if force_callout else candidates[::max(1, len(candidates) // 800)]):
            point = points[candidate]
            px, py = ax.transData.transform(point)
            trial = Bbox.from_extents(px - width / 2 - 1.2, py - height / 2 - 1.8,
                                      px + width / 2 + 1.2, py + height / 2 + 1.8)
            if (ax.bbox.contains(trial.x0, trial.y0) and ax.bbox.contains(trial.x1, trial.y1)
                    and not any(trial.overlaps(other) for other in blocked)):
                position, box = point, trial
                break
        if position is None:
            target = points[candidates[0]]
            if force_callout:
                core = points[(points[:, 0] > -.95) & (points[:, 0] < -.45)
                              & (abs(points[:, 1]) < .25)]
                if len(core):
                    target = core[np.argmin(core[:, 0])]
            locations = ((-2., .5), (-2., -.5), (-1.9, 1.65)) if force_callout else (
                (0., 1.65), (1.5, 1.65), (-1.5, -1.6), (0., -1.5))
            for location in locations:
                px, py = ax.transData.transform(location)
                box = Bbox.from_extents(px - width / 2 - 1.2, py - height / 2 - 1.8,
                                       px + width / 2 + 1.2, py + height / 2 + 1.8)
                if (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                        and not any(box.overlaps(other) for other in blocked)):
                    ax.annotate(f"{level:g}", xy=target, xytext=location,
                                fontsize=6., ha="center", va="center", zorder=7,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=.15),
                                arrowprops=dict(arrowstyle="->", color="#555555", lw=.45,
                                                mutation_scale=4.5))
                    labels.append(dict(value=float(level), position=list(location),
                                       contour_position=target.tolist(), placement="callout"))
                    blocked.append(box)
                    break
            else:
                raise ValueError(f"No clear contour label location {kind} {level}")
            continue
        count = len(contours.labelTexts)
        # Matplotlib needs the full level list when the location is manual.
        ax.clabel(contours, levels=contours.levels, manual=[position], inline=True,
                  fontsize=6., fmt=lambda val: f"{val:g}", inline_spacing=1.)
        for text in contours.labelTexts[count:]:
            text.set_fontweight("bold")
            text.set_rotation(0)
            text.set_bbox(dict(facecolor="white", edgecolor="none", alpha=.86, pad=.10))
            labels.append(dict(value=float(text.get_text()), position=list(text.get_position()),
                               placement="inline"))
            blocked.append(box)
    return labels


def build_figure(plane: str, target: Path) -> dict:
    """Write one single-column figure and return data/layout audit metadata."""
    if plane not in ("xy", "xz"):
        raise ValueError(plane)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in (SOURCE, GEOMETRY)}
    with np.load(SOURCE) as saved:
        data = {key: saved[key] for key in saved.files}
    geometry = json.loads(GEOMETRY.read_text())
    x, vertical = data["x_angstrom"], data["vertical_angstrom"]
    dx, dv = x[1] - x[0], vertical[1] - vertical[0]
    extent = [x[0] - dx / 2, x[-1] + dx / 2, vertical[0] - dv / 2, vertical[-1] + dv / 2]
    reference = data[f"{plane}_cc-pvtz"]
    width, height = FIGURE_INCHES
    lefts = (.34, 1.91)
    plot_width = 1.35
    plot_height = plot_width * (vertical[-1] - vertical[0]) / (x[-1] - x[0])
    bottoms = (4.15, 2.55, .95)
    panels = []
    style = {"font.family": "DejaVu Sans", "font.weight": "bold",
             "axes.labelweight": "bold", "mathtext.fontset": "dejavusans",
             "mathtext.default": "bf", "font.size": 7., "axes.labelsize": 7.2,
             "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
             "axes.linewidth": .7, "xtick.direction": "in", "ytick.direction": "in",
             "xtick.major.width": .55, "ytick.major.width": .55,
             "xtick.major.size": 2.3, "ytick.major.size": 2.3,
             "pdf.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        figure = plt.figure(figsize=FIGURE_INCHES)
        figure.text(.98, .988, r"$xy,\ z=0$" if plane == "xy" else r"$xz,\ y=0$",
                    ha="right", va="top", fontsize=7.3)
        for col, title in enumerate((r"$\rho$", r"$\rho-\rho_{\mathrm{cc-pVTZ}}$")):
            figure.text((lefts[col] + plot_width / 2) / width, 5.56 / height,
                        title, ha="center", va="center", fontsize=8.2)
        for row, ((key, label), bottom) in enumerate(zip(METHODS, bottoms)):
            rho = data[f"{plane}_{key}"]
            delta = rho - reference
            assert rho.shape == delta.shape == (561, 641)
            assert np.all(np.isfinite(rho)) and np.all(rho >= 0) and np.all(np.isfinite(delta))
            for col, left in enumerate(lefts):
                ax = figure.add_axes([left / width, bottom / height,
                                      plot_width / width, plot_height / height])
                field = rho if col == 0 else delta
                image = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                                  interpolation="nearest", cmap="Blues" if col == 0 else "RdBu_r",
                                  norm=Normalize(0., 1.5) if col == 0 else Normalize(-.1, .1))
                assert np.array_equal(np.asarray(image.get_array()), field)
                levels = CONTOURS if col == 0 else DIFFERENCE_CONTOURS
                contours = ax.contour(x, vertical, field, levels=levels, colors="#606060",
                                      linewidths=.45, alpha=.95)
                if col == 0:
                    density_image = image
                else:
                    difference_image = image
                marked = nuclei(ax, geometry, plane)
                ax.set(xlim=(x[0], x[-1]), ylim=(vertical[0], vertical[-1]),
                       xticks=[-2, 0, 2], yticks=[-2, 0, 2])
                ax.tick_params(top=True, right=True, labelbottom=row == 2,
                               labelleft=col == 0, pad=2.)
                if col == 0:
                    ax.set_ylabel(("y" if plane == "xy" else "z") + " (Å)", labelpad=2.)
                tag = f"({chr(97 + 2 * row + col)})" + (" " + label if col == 0 else "")
                ax.text(.025, .975, tag, transform=ax.transAxes,
                        fontsize=6.5, ha="left", va="top", zorder=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=.15))
                labels = contour_labels(ax, contours, "density" if col == 0 else "difference")
                # Numeric extrema remain outside the data window at their true values.
                ext_top = bottom - (.21 if row == 2 else .08)
                for line, (name, val) in enumerate((("min", field.min()), ("max", field.max()))):
                    figure.text(left / width, (ext_top - line * .12) / height,
                                rf"$\{name}={value_label(float(val))}$", fontsize=6.2,
                                ha="left", va="top")
                panels.append(dict(row=row, column=col, method=key,
                    source_array_key=f"{plane}_{key}",
                    reference_array_key=f"{plane}_cc-pvtz" if col else None,
                    shape=list(field.shape), minimum=float(field.min()), maximum=float(field.max()),
                    displayed_array_sha256=digest(field), source_values_preserved=True,
                    nuclei_indices=marked, labeled_contours=labels,
                    display_limits=[0., 1.5] if col == 0 else [-.1, .1]))
        figure.text(.53, .445 / height, "x along C=C (Å)", fontsize=7.2, ha="center", va="center")
        for col, (im, label, extend, ticks) in enumerate((
                (density_image, r"$\rho$ ($e\,a_0^{-3}$)", "max", [0., .5, 1., 1.5]),
                (difference_image, r"$\Delta\rho$ ($e\,a_0^{-3}$)", "both", [-.1, 0., .1]))):
            cax = figure.add_axes([lefts[col] / width, .285 / height, plot_width / width, .065 / height])
            bar = figure.colorbar(im, cax=cax, orientation="horizontal", extend=extend)
            bar.set_ticks(ticks)
            bar.ax.tick_params(labelsize=6., length=2., width=.5, pad=1.8)
            bar.outline.set_linewidth(.6)
            bar.set_label(label, fontsize=7., labelpad=2.)
        figure.savefig(target, dpi=600, facecolor="white")
        png = PREVIEWS / (target.stem + ".png")
        figure.savefig(png, dpi=400, facecolor="white")
        plt.close(figure)
    with fitz.open(target) as pdf:
        assert len(pdf) == 1
        page = pdf[0]
        out_of_bounds = []
        sizes = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes.append(span["size"])
                    if not page.rect.contains(fitz.Rect(span["bbox"])):
                        out_of_bounds.append(dict(text=span["text"], bbox=span["bbox"]))
        assert not out_of_bounds, out_of_bounds
        page_points = list(page.rect)
    assert all(sha(ROOT / path) == value for path, value in hashes.items())
    record = dict(plane=plane, target=str(target.relative_to(ROOT)), figure_inches=list(FIGURE_INCHES),
                  panels=panels, page_points=page_points, text_outside_page=out_of_bounds,
                  contour_font_pt=6., extrema_font_pt=6.2, axes_font_pt=7.2,
                  color_scales="Linear; shared density [0,1.5] and difference [-0.1,0.1] electron/bohr^3",
                  interpolation="nearest; no smoothing", data_window_preserved=True,
                  density_contours=CONTOURS, difference_contours=DIFFERENCE_CONTOURS,
                  source_sha256=hashes, sources_unchanged=True, target_sha256=sha(target),
                  preview=str(png.relative_to(ROOT)), script_sha256=sha(Path(__file__)),
                  scope="Static plotting of saved total-density fields only; no physical calculation.")
    (PREVIEWS / (target.stem + "-report.json")).write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    records = [build_figure(plane, ROOT / f"manuscript/figures/fig-ethylene-density-{plane}.pdf")
               for plane in ("xy", "xz")]
    (PREVIEWS / "report.json").write_text(json.dumps(dict(figures=records), indent=2) + "\n")
    print(json.dumps(dict(figures=2, width_inches=FIGURE_INCHES[0],
                          report=str((PREVIEWS / "report.json").relative_to(ROOT)))))


if __name__ == "__main__":
    main()
