"""Replot the three training statistics as a half-slide vertical triptych."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from plot_case08_reactions import ROOT, configure_style, sha
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator
import numpy as np


OUT = ROOT / "output/ppt/training_statistics"
STEM = "training-statistics-triptych"
BLUE = "#0057B8"


def read_csv(name):
    with (OUT / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    configure_style()
    elements = read_csv("element-distribution.csv")
    heavy = read_csv("heavy-atom-distribution.csv")
    bins = read_csv("ao-histogram-bins.csv")
    assert len(elements) == 25 and sum(int(r["atom_count"]) for r in elements) == 5402
    assert sum(int(r["structures"]) for r in heavy) == sum(int(r["structures"]) for r in bins) == 500
    with plt.rc_context({"font.weight": "bold", "axes.labelweight": "bold",
                         "axes.labelsize": 14, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "axes.linewidth": 1.5,
                         "xtick.major.size": 4, "ytick.major.size": 4,
                         "xtick.major.width": 1.2, "ytick.major.width": 1.2}):
        figure = plt.figure(figsize=(6.2, 6.6))
        axes = [figure.add_axes([.15, bottom, .82, .21]) for bottom in (.75, .43, .11)]
        for tag, ax in zip(("(a)", "(b)", "(c)"), axes):
            ax.tick_params(top=True, right=True)
            ax.text(-.145, 1.04, tag, transform=ax.transAxes, fontsize=12,
                    ha="left", va="bottom", fontweight="bold")

        ax = axes[0]
        ax.bar(np.arange(len(elements)), [int(r["atom_count"]) for r in elements],
               color=BLUE, width=.74, zorder=3)
        # Two staggered rows keep all 25 chemical symbols horizontal and legible.
        labels = [("\n" if i % 2 else "") + r["element"] for i, r in enumerate(elements)]
        ax.set_xticks(np.arange(len(elements)), labels, fontsize=12)
        ax.set_xlim(-.8, len(elements) - .2)
        ax.set_yscale("log")
        ax.set_ylim(.7, 8000)
        ax.set_yticks([1, 10, 100, 1000])
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2., 5.), numticks=100))
        ax.tick_params(which="minor", top=True, right=True, length=2, width=.8)
        ax.set_ylabel("Atoms", labelpad=5)
        ax.text(.98, .95, "5,402 atoms", transform=ax.transAxes, fontsize=11,
                ha="right", va="top")

        ax = axes[1]
        ax.bar([int(r["heavy_atoms"]) for r in heavy], [int(r["structures"]) for r in heavy],
               color=BLUE, width=.86)
        ax.set_xlim(-.75, 20.75)
        ax.set_ylim(0, 96)
        ax.set_xticks(range(0, 21, 2))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.set_xlabel("Heavy atoms", labelpad=4)
        ax.set_ylabel("Structures", labelpad=5)
        ax.text(.98, .95, "500 structures", transform=ax.transAxes, fontsize=11,
                ha="right", va="top")

        ax = axes[2]
        ax.bar([int(r["left_inclusive"]) for r in bins], [int(r["structures"]) for r in bins],
               align="edge", width=9.6, color=BLUE)
        ax.set_xlim(0, 250)
        ax.set_ylim(0, 60)
        ax.set_xticks(range(0, 251, 50))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.set_xlabel("AOs (NNAO-SZP-L)", labelpad=4)
        ax.set_ylabel("Structures", labelpad=5)
        ax.text(.98, .95, "500 structures", transform=ax.transAxes, fontsize=11,
                ha="right", va="top")
        for extension in ("png", "pdf", "svg"):
            figure.savefig(OUT / (STEM + "." + extension), dpi=320)
        plt.close(figure)
    report = dict(figure_inches=[6.2, 6.6], figure_cm=[15.748, 16.764], png_dpi=320,
                  intended_slide_inches=[13.3333333333, 7.5],
                  fits_half_slide_width=True, panel_order=["elements", "heavy atoms", "NNAO-SZP-L AOs"],
                  element_axis="Logarithmic y; two staggered rows of horizontal element labels.",
                  axis_label_pt=14, tick_label_pt=12, panel_label_pt=12, annotation_pt=11,
                  font_weight="bold", frame_width_pt=1.5, plotted_totals=[5402, 500, 500],
                  source_sha256={str(OUT / name): sha(OUT / name) for name in
                      ["element-distribution.csv", "heavy-atom-distribution.csv", "ao-histogram-bins.csv"]},
                  script_sha256=sha(Path(__file__)))
    (OUT / (STEM + "-report.json")).write_text(json.dumps(report, indent=2) + "\n")
    path = OUT / "index.html"
    content = path.read_text()
    section = '<section id="triptych"><h2>Half-slide vertical triptych</h2>' + \
        '<a href="' + STEM + '.png"><img src="' + STEM + '.png" alt="Three vertically stacked training statistics" style="max-width:650px"></a><p>' + \
        ' | '.join('<a href="' + STEM + '.' + ext + '">' + ext.upper() + '</a>' for ext in ("png", "pdf", "svg")) + '</p></section>'
    if '<section id="triptych">' in content:
        start = content.index('<section id="triptych">')
        end = content.index('</section>', start) + len('</section>')
        content = content[:start] + section + content[end:]
    else:
        content = content.replace('<section>', section + '<section>', 1)
    path.write_text(content)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
