"""Plot archived C0 contraction changes, pairing MACE and free contraction.

Use --ppt. Each scan has two half-slide s/p/d triptychs with matched y limits.
Raw coefficients use the source unit-vector gauge; changes use each method's
own initial geometry. No electronic-structure calculations are performed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from plot_case08_reactions import ROOT, configure_style, sha
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
import fitz


CASE = ROOT / "data/computational_examples/2026-10-27/cases/16_ethane_spd_radial_response"
SOURCE = CASE / "tables/primitive-coefficients.csv"
OUT = ROOT / "output/ppt/ethane_contraction_changes"
COLORS = ["#246694", "#C38B1B", "#CB602B", "#71813A", "#AE5B80", "#765F99"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
METHODS = {"MACE": "MACE", "Free": "Free contraction"}


def get_series(rows, scan, channel, method):
    selected = [r for r in rows if r["scan"] == scan and r["channel"] == channel and r["method"] == method]
    indices = sorted({int(r["point_index"]) for r in selected})
    count = 3 if channel == "d" else 6
    by_key = {(int(r["point_index"]), int(r["primitive"])): r for r in selected}
    assert len(by_key) == len(selected) == len(indices) * count
    x = np.array([float(by_key[i, 1]["coordinate"]) for i in indices])
    raw = np.array([[float(by_key[i, j]["raw_coefficient"]) for j in range(1, count + 1)] for i in indices])
    delta = np.array([[float(by_key[i, j]["raw_change_from_scan_start"]) for j in range(1, count + 1)] for i in indices])
    exponents = [float(by_key[indices[0], j]["exponent_bohr_minus2"]) for j in range(1, count + 1)]
    assert np.array_equal(delta, raw - raw[0])
    assert np.all(np.diff(exponents) < 0)
    assert np.allclose(np.linalg.norm(raw, axis=1), 1., atol=1e-12, rtol=0.)
    return x, raw, delta, exponents


def draw_scan(rows, scan):
    scale = 1000. if scan == "torsion" else 1.
    series, limits = {}, {}
    for channel in "spd":
        for method in METHODS:
            series[channel, method] = get_series(rows, scan, channel, method)
        assert np.array_equal(series[channel, "MACE"][0], series[channel, "Free"][0])
        assert series[channel, "MACE"][3] == series[channel, "Free"][3]
        values = np.concatenate([scale * series[channel, method][2].ravel() for method in METHODS])
        low, high = float(values.min()), float(values.max())
        span = high - low
        limits[channel] = [low - .12 * span, high + (.65 if channel != "d" else .40) * span]
    figures, exported = [], []
    for method, title in METHODS.items():
        figure = plt.figure(figsize=(6.2, 6.6))
        figure.text(.97, .99, title, fontsize=14, fontweight="bold", ha="right", va="top")
        axes = [figure.add_axes([.16, bottom, .81, .225]) for bottom in (.72, .415, .11)]
        panels = []
        for channel, tag, ax in zip("spd", ("(a)", "(b)", "(c)"), axes):
            x, raw, delta, exponents = series[channel, method]
            ax.axhline(0., color=".55", lw=.8, zorder=0)
            for j in range(delta.shape[1]):
                ax.plot(x, scale * delta[:, j], color=COLORS[j],
                        ls="-" if method == "MACE" else "--", lw=2.3,
                        marker=MARKERS[j], ms=3.1, markeredgewidth=.7,
                        markerfacecolor=COLORS[j] if method == "MACE" else "white",
                        markevery=2 if scan == "torsion" else 4,
                        label=rf"$c_{{{j + 1}}}$", zorder=3)
                exported.extend(dict(scan=scan, method=method, channel=channel,
                    point_index=i, coordinate=float(coordinate), primitive=j + 1,
                    exponent_bohr_minus2=exponents[j], raw_coefficient=float(raw[i, j]),
                    change_from_own_start=float(delta[i, j]), plotted_value=float(scale * delta[i, j]),
                    display_multiplier=scale) for i, coordinate in enumerate(x))
            ax.set_ylim(limits[channel])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.set_ylabel(r"$10^3\Delta c_i$" if scan == "torsion" else r"$\Delta c_i$", labelpad=6)
            ax.tick_params(top=True, right=True)
            ax.text(-.16, 1.04, tag + " " + rf"${channel}$", transform=ax.transAxes,
                    fontsize=12, fontweight="bold", ha="left", va="bottom")
            handles, labels = ax.get_legend_handles_labels()
            order = [0, 3, 1, 4, 2, 5] if len(handles) == 6 else list(range(len(handles)))
            legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                              loc="upper center", ncol=3, fontsize=11, handlelength=1.6,
                              handletextpad=.4, columnspacing=1., labelspacing=.15, borderaxespad=.35)
            if scan == "torsion":
                ax.set_xlim(-3., 123.)
                ax.set_xticks([0, 30, 60, 90, 120])
                xlabel = r"Methyl rotation $\phi$ (deg)"
            else:
                ax.set_xlim(x[0] - .10, x[-1] + .12)
                ax.set_xticks([2, 3, 4, 5, 6, 7, 8])
                xlabel = r"C-C distance $R$ (Å)"
            if channel == "d":
                ax.set_xlabel(xlabel, labelpad=4)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            # The legend occupies reserved headroom above all coefficient data.
            legend_low = ax.transData.inverted().transform((0., legend.get_window_extent(renderer).y0))[1]
            assert legend_low > float((scale * delta).max()), (scan, method, channel)
            panels.append(dict(channel=channel, points=len(x), coefficients=len(exponents),
                               exponents_bohr_minus2=exponents, matched_y_limits=limits[channel],
                               legend_bottom_data_y=float(legend_low)))
        stem = f"ethane-{scan}-{method.lower()}-coefficient-triptych"
        for extension in ("png", "pdf", "svg"):
            figure.savefig(OUT / (stem + "." + extension), dpi=320)
        plt.close(figure)
        figures.append(dict(scan=scan, method=method, title=title, stem=stem,
                            display_multiplier=scale, figure_inches=[6.2, 6.6], panels=panels))
    pair_stem = f"ethane-{scan}-coefficient-comparison"
    paired = Image.new("RGB", (3968, 2112), "white")
    pdf = fitz.open()
    page = pdf.new_page(width=12.4 * 72, height=6.6 * 72)
    for i, metadata in enumerate(figures):
        stem = metadata["stem"]
        with Image.open(OUT / (stem + ".png")) as image:
            paired.paste(image.convert("RGB"), (1984 * i, 0))
        with fitz.open(OUT / (stem + ".pdf")) as source:
            page.show_pdf_page(fitz.Rect(i * 6.2 * 72, 0, (i + 1) * 6.2 * 72, 6.6 * 72), source, 0)
    paired.save(OUT / (pair_stem + ".png"))
    pdf.save(OUT / (pair_stem + ".pdf"))
    pdf.close()
    preview = Image.new("RGB", (1280, 720), "white")
    for i, metadata in enumerate(figures):
        with Image.open(OUT / (metadata["stem"] + ".png")) as image:
            preview.paste(image.convert("RGB").resize((595, 634), Image.Resampling.LANCZOS), (22 + 640 * i, 43))
    preview.save(OUT / (pair_stem + "-slide-preview.png"))
    return exported, figures, dict(scan=scan, stem=pair_stem, methods=["MACE", "Free"],
                                    figure_inches=[12.4, 6.6])


def main():
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2430 and all(r["atom_label"] == "C0" for r in rows)
    with plt.rc_context({"font.weight": "bold", "axes.labelweight": "bold", "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.5,
                         "xtick.major.size": 4, "ytick.major.size": 4,
                         "xtick.major.width": 1.2, "ytick.major.width": 1.2,
                         "mathtext.default": "bf"}):
        exported, figures, pairs = [], [], []
        for scan in ("stretch", "torsion"):
            records, half_figures, pair = draw_scan(rows, scan)
            exported.extend(records)
            figures.extend(half_figures)
            pairs.append(pair)
    assert len(exported) == 2430
    with (OUT / "plotted-coefficient-changes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    report = dict(figures=figures, paired_figures=pairs, plotted_values=len(exported),
        basis="Historical update-3351 NNAO, 44-AO ethane", atom_index=0,
        definition="raw c_i(q) - raw c_i(q0), each method uses its own scan-start coefficients",
        gauge="Source raw contraction columns have Euclidean norm 1; source phase is retained.",
        primitive_order="Descending exponent, tight to diffuse; shared color/marker map for both methods.",
        display_scaling=dict(stretch=1, torsion=1000), y_limits="Matched MACE/Free within each scan and channel.",
        source_sha256={str(p): sha(p) for p in [SOURCE, CASE / "arrays/radial-functions.npz",
            CASE / "README.md", ROOT / "scripts/plot_case08_reactions.py", Path(__file__)]},
        scope="Archived coefficients only. Fixed core excluded; no network inference or SCF. "
              "Changes from different method-specific origins do not show their initial absolute coefficient gap.")
    (OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    cards = []
    for pair in pairs:
        stem = pair["stem"]
        title = "C-C stretching" if pair["scan"] == "stretch" else "Methyl rotation"
        links = []
        for figure in (f for f in figures if f["scan"] == pair["scan"]):
            links.append(figure["title"] + ': ' + ' | '.join('<a href="' + figure["stem"] + '.' + ext + '">' +
                         ext.upper() + '</a>' for ext in ("png", "pdf", "svg")))
        cards.append('<section><h2>' + title + '</h2><a href="' + stem + '.png"><img src="' + stem +
                     '.png" alt="' + title + ': MACE and free contraction coefficient changes"></a><p>' +
                     '<a href="' + stem + '.pdf">Paired PDF</a></p><p>' + '</p><p>'.join(links) + '</p></section>')
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Ethane contraction changes</title><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{max-width:1450px;margin:30px auto;padding:0 20px;font:16px/1.5 Arial;color:#171717}'
        'img{width:100%;height:auto}a{color:#0057b8}section{margin:35px 0}</style>'
        '<h1>Ethane C0 contraction coefficient changes</h1><p>Each pair: MACE on the left, free contraction on the right. '
        'Rows: valence s, valence p, polarization d. Coefficients are ordered from tight to diffuse; '
        'color and marker identities are shared. Each method uses its own initial coefficients. '
        'The rotation y axis explicitly shows 1000 times the coefficient change.</p>' + ''.join(cards) + '</html>')
    print(json.dumps({k: v for k, v in report.items() if k not in ("figures", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
