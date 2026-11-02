"""Plot MACE-only ethane C0 radial functions as two half-slide triptychs.

Use --ppt. The archived arrays already contain normalized, phase-aligned
R_l(r); the plotted reduced radial function is u_l(r)=r R_l(r).
"""
from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path

from plot_case08_reactions import ROOT, configure_style, sha
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter


CASE = ROOT / "data/computational_examples/2026-10-27/cases/16_ethane_spd_radial_response"
ARRAYS = CASE / "arrays/radial-functions.npz"
OUT = ROOT / "output/ppt/ethane_mace_radial"
SNAPSHOTS = [("#171717", "-", 2.4), ("#0057B8", "--", 2.8), ("#ef233c", ":", 2.8)]


def legend_labels(scan, coordinates):
    if scan == "torsion":
        return [rf"$\phi={value:.0f}^\circ$" for value in coordinates]
    return [rf"$R={value:.3f}$ Å" if i == 0 else rf"$R={value:.1f}$ Å"
            for i, value in enumerate(coordinates)]


def draw_scan(data, scan):
    radius = data["radius_bohr"]
    coordinate = data[scan + "_coordinate"]
    targets = [0., 30., 60.] if scan == "torsion" else [coordinate[0], 3., 8.]
    indices = [int(np.argmin(abs(coordinate - value))) for value in targets]
    selected = coordinate[indices]
    labels = legend_labels(scan, selected)
    shown = radius <= 6.
    records, panels = [], []
    figure = plt.figure(figsize=(6.2, 6.6))
    axes = [figure.add_axes([.16, bottom, .81, .23]) for bottom in (.73, .42, .11)]
    for channel, tag, ax in zip("spd", ("(a)", "(b)", "(c)"), axes):
        key = f"{scan}_mace_{channel}_R"
        functions = data[key][indices] * radius
        ax.axhline(0., color=".55", lw=.8, zorder=0)
        for i, (color, linestyle, linewidth) in enumerate(SNAPSHOTS):
            ax.plot(radius[shown], functions[i, shown], color=color, ls=linestyle,
                    lw=linewidth, zorder=3, label=labels[i])
            records.extend(dict(scan=scan, channel=channel, method="MACE",
                geometry_coordinate=float(selected[i]), radius_bohr=float(r),
                u_bohr_minus_half=float(u)) for r, u in zip(radius[shown], functions[i, shown]))
        ax.set_xlim(0., 6.)
        ax.set_ylim(-.08, 1.08)
        ax.set_xticks(range(7))
        ax.set_yticks([0., .5, 1.])
        ax.set_ylabel(rf"$u_{channel}(r)$ ($a_0^{{-1/2}}$)", labelpad=5)
        ax.tick_params(top=True, right=True)
        ax.text(-.16, 1.04, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="left", va="bottom")
        if channel == "s":
            ax.legend(loc="upper right", ncol=3, fontsize=11, handlelength=1.6,
                      handletextpad=.4, columnspacing=.8, borderaxespad=.4)
        if channel == "d":
            ax.set_xlabel(r"Radius from C0 ($a_0$)", labelpad=4)
        metadata = dict(channel=channel, array_key=key, geometry_indices=indices,
                        geometry_coordinates=selected.tolist(), radial_points=int(shown.sum()),
                        y_limits=[-.08, 1.08], phase="Preserved from archived arrays")
        if scan == "torsion":
            inset = ax.inset_axes([.59, .38, .37, .36])
            inset.axhline(0., color=".55", lw=.6, zorder=0)
            for i in (1, 2):
                color, linestyle, _ = SNAPSHOTS[i]
                difference = 1000 * (functions[i] - functions[0])
                inset.plot(radius[shown], difference[shown], color=color, ls=linestyle, lw=1.6)
            inset.set_xlim(0., 6.)
            inset.set_ylim(-6.5, 6.5)
            inset.set_xticks([0., 3., 6.])
            inset.set_yticks([-5., 0., 5.])
            inset.tick_params(labelsize=9, direction="in", top=True, right=True,
                              length=2, width=.8, pad=1.5)
            for spine in inset.spines.values():
                spine.set_linewidth(1.)
            inset.text(.97, .94, r"$10^3\Delta u$", transform=inset.transAxes,
                       fontsize=9, ha="right", va="top")
            metadata["inset"] = dict(quantity="1000 * (u(phi) - u(0))",
                                     angles_deg=selected[1:].tolist(), y_limits=[-6.5, 6.5],
                                     source_difference_scaling=1000.)
        panels.append(metadata)
    stem = f"ethane-{scan}-mace-radial-triptych"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUT / (stem + "." + extension), dpi=320)
    plt.close(figure)
    return records, dict(scan=scan, stem=stem, panels=panels, legend_labels=labels,
                        figure_inches=[6.2, 6.6], png_dpi=320,
                        radius_display_bohr=[0., 6.], main_y_limits=[-.08, 1.08])


def main():
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ARRAYS) as data, plt.rc_context({
        "font.weight": "bold", "axes.labelweight": "bold", "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.5,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "mathtext.default": "bf",
    }):
        records, figures = [], []
        for scan in ("stretch", "torsion"):
            points, metadata = draw_scan(data, scan)
            records.extend(points)
            figures.append(metadata)
    with (OUT / "plotted-radial-values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    book = PdfWriter()
    cards = []
    for metadata in figures:
        stem = metadata["stem"]
        title = "C-C stretching" if metadata["scan"] == "stretch" else "Methyl rotation"
        book.append(PdfReader(OUT / (stem + ".pdf")))
        cards.append('<section><h2>' + title + '</h2><a href="' + stem + '.png">'
            '<img src="' + stem + '.png" alt="' + escape(title) + ': MACE-only carbon s/p/d radial functions"></a><p>' +
            ' | '.join('<a href="' + stem + '.' + extension + '">' + extension.upper() + '</a>'
                       for extension in ("png", "pdf", "svg")) + '</p></section>')
    book.write(OUT / "ethane-mace-radial-triptychs.pdf")
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>MACE radial functions</title>'
        '<style>body{max-width:1300px;margin:30px auto;padding:0 20px;font:16px/1.5 Arial;color:#171717}'
        '.grid{display:grid;grid-template-columns:1fr 1fr;gap:25px}img{width:100%;height:auto}'
        'a{color:#0057b8}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>'
        '<h1>Ethane MACE radial functions</h1><p>Carbon C0 active s, p and d basis radial functions. '
        'Only the archived update-3351 MACE predictions are shown. The fixed core is excluded. '
        'Each triptych is 6.2 by 6.6 inches for one half of a 16:9 slide.</p>'
        '<p>The main axes show u(r)=rR(r), with the original normalization and phase. '
        'Rotation insets show 1000 times the change from the 0-degree curve; '
        'their small response is not rescaled in the main plots.</p><div class="grid">' + ''.join(cards) + '</div></html>')
    report = dict(figures=figures, method="MACE only", checkpoint="historical update 3351", atom="C0 (index 0)",
        channel_roles={"s": "valence 6-to-1", "p": "valence 6-to-1", "d": "polarization 3-to-1"},
        shell_indices_zero_based={"s": 1, "p": 2, "d": 3}, fixed_core_shell_excluded=0,
        radial_definition="u_l(r) = r * archived R_l(r)",
        units=dict(radius="bohr", u="bohr^(-1/2)", stretch_coordinate="Angstrom", torsion_coordinate="degree"),
        main_curve_points=len(records), normalization="Archived full-domain normalization; no renormalization or peak scaling.",
        font_weight="bold", axis_label_pt=14, tick_label_pt=12, frame_width_pt=1.5,
        source_sha256={str(p): sha(p) for p in [ARRAYS, CASE / "README.md", CASE / "verification.json",
            CASE / "input-manifest.json", ROOT / "scripts/plot_case08_reactions.py", Path(__file__)]})
    (OUT / "plot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("figures", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
