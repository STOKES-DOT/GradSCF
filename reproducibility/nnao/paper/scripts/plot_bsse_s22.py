"""Draw separate, matched-cohort S22 BSSE summaries from archived means."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


COHORTS = [
    ("full_22", "S22: all 22 dimers", "s22-bsse-full-22", 22, 7),
    ("common_12", "S22: common 12 dimers", "s22-bsse-common-12", 12, 8),
]


def _panel(ax, selected, series, title, limits):
    shown = [item for item in series if item[0] in selected]
    for position, (basis, _, color, _, _) in enumerate(shown):
        value = float(selected[basis]["mean_bsse_kcal_mol"])
        assert value > 0
        size = 105 if basis == "cc-pvtz" else 82
        marker = "s" if basis == "NNAO/MACE" else "o"
        ax.scatter(value, position, color=color, s=size, marker=marker,
                   edgecolors=color, linewidths=1., zorder=3)
        ax.annotate(f"{value:.4f}", (value, position), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=12.5)
    ax.set_yticks(range(len(shown)), [item[1] for item in shown], fontsize=12.5)
    ax.set_ylim(len(shown) - .35, -.65)
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    ax.set_xticks([.01, .1, 1., 10., 100.])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2., 5.)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", top=True, right=True)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(which="minor", length=2.5, width=.8)
    ax.set_xlabel(r"Mean BSSE (kcal mol$^{-1}$)")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)


def _save(figure, path):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = figure.bbox
    for text in figure.findobj(plt.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1), (
                path.name, text.get_text(), box.bounds)
    for ax in figure.axes:
        for annotation in ax.texts:
            box = annotation.get_window_extent(renderer)
            assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1), (
                path.name, annotation.get_text(), box.bounds)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(str(path) + "." + suffix, dpi=320)
    plt.close(figure)


def draw_s22(case: Path, out: Path, series: list) -> tuple[list[dict], list[dict]]:
    """Render two square cohort figures and one matched-scale comparison."""
    with (case / "tables/s22-summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15
    out.mkdir(parents=True, exist_ok=True)
    # Shared limits retain all means and reserve space for their value labels.
    limits = (.01, 600.)
    selections, plotted, figures = {}, [], []
    for cohort, title, stem, count, method_count in COHORTS:
        selected = {row["basis"]: row for row in rows if row["cohort"] == cohort}
        assert len(selected) == method_count
        assert all(int(row["n"]) == count and len(row["structure_ids"].split(";")) == count
                   for row in selected.values())
        selections[cohort] = selected
        figure = plt.figure(figsize=(6.3, 6.3))
        ax = figure.add_axes([.36, .16, .60, .73])
        _panel(ax, selected, series, title, limits)
        _save(figure, out / stem)
        for basis, _, _, _, _ in series:
            if basis in selected:
                row = selected[basis]
                plotted.append(dict(figure=stem, cohort=cohort, basis=basis, n=count,
                                    value_kcal_mol=float(row["mean_bsse_kcal_mol"])))
        figures.append(dict(stem=stem, title=title, cohort=cohort, is_summary=False,
                            x_scale="log", x_limits=list(limits), text_within_figure=True,
                            figure_inches=[6.3, 6.3]))
    assert len(plotted) == 15
    figure = plt.figure(figsize=(12.6, 6.3))
    for (cohort, title, _, _, _), left in zip(COHORTS, (.18, .68)):
        ax = figure.add_axes([left, .16, .30, .73])
        _panel(ax, selections[cohort], series, title, limits)
    _save(figure, out / "s22-bsse-summary")
    figures.append(dict(stem="s22-bsse-summary", title="S22 BSSE: matched-cohort comparisons",
                        cohort="full_22 / common_12", is_summary=True, x_scale="log",
                        x_limits=list(limits), text_within_figure=True,
                        figure_inches=[12.6, 6.3]))
    return plotted, figures
