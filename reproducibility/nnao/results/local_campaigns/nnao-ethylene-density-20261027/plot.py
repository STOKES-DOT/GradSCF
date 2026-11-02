"""Replot the two six-panel density maps using saved numerical arrays only."""
import json
import math
import os
from pathlib import Path

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nnao-density-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = Path(__file__).resolve().parent
KEYS = ["mace", "free", "sto6g", "sto6g-p", "cc-pvtz", "cc-pvdz"]
LABELS = ["NNAO / MACE", "NNAO / free contraction", "STO-6G", "STO-6G+P", "cc-pVTZ", "cc-pVDZ"]


def main():
    results = json.loads((OUT / "results.json").read_text())
    assert results["status"] == "complete" and results["passed"] == 6
    records = {r["key"]: r for r in results["records"]}
    geometry = json.loads((OUT / "inputs/structure.json").read_text())
    coords = np.asarray(geometry["coords_angstrom"])
    with np.load(OUT / "arrays/plane-densities.npz") as saved:
        data = {k: np.array(saved[k]) for k in saved.files}
    x, vertical = data["x_angstrom"], data["vertical_angstrom"]
    maximum = max(float(data[f"{plane}_{key}"].max()) for plane in ("xy", "xz") for key in KEYS)
    vmin, vmax = 1e-4, 25. * math.ceil(maximum / 25.)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["Blues"].with_extremes(under="white")
    levels = [.005, .02, .08, .32, 1.28]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
        "axes.linewidth": .8, "xtick.direction": "in", "ytick.direction": "in"})

    def panel(ax, key, plane, title):
        rho = data[f"{plane}_{key}"]
        dx, dy = x[1] - x[0], vertical[1] - vertical[0]
        im = ax.imshow(rho, origin="lower",
                       extent=[x[0] - dx/2, x[-1] + dx/2, vertical[0] - dy/2, vertical[-1] + dy/2],
                       cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        lines = ax.contour(x, vertical, rho, levels=levels, colors="#777E85", linewidths=.5)
        ax.clabel(lines, levels=[.005, .08, .32], fmt="%.3g", fontsize=7, inline_spacing=3)
        axis = 1 if plane == "xy" else 2
        omitted_axis = 2 if plane == "xy" else 1
        for symbol, center in zip(geometry["symbols"], coords):
            if abs(center[omitted_axis]) > 1e-10:
                continue
            xx, yy = center[0], center[axis]
            ax.scatter([xx], [yy], s=11, color="#14191E", edgecolors="white", linewidths=.5, zorder=5)
            ax.annotate(symbol, (xx, yy), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=9, color="#1D2329", zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=.4))
        ax.set_title(title, fontsize=11, loc="left", pad=8)
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(vertical[0], vertical[-1])
        ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-2, -1, 0, 1, 2])
        return im

    (OUT / "figures").mkdir(exist_ok=True)
    outputs = []
    for plane in ("xy", "xz"):
        fig, axes = plt.subplots(3, 2, figsize=(11.1, 13.0), sharex=True, sharey=True,
                                 layout="constrained")
        fig.get_layout_engine().set(h_pad=.10, w_pad=.08, hspace=.04, wspace=.02)
        for i, (ax, key, label) in enumerate(zip(axes.flat, KEYS, LABELS)):
            im = panel(ax, key, plane, f"({chr(97+i)}) {label} · {records[key]['nao']} AO")
            if i % 2 == 0:
                ax.set_ylabel(f"{'y' if plane == 'xy' else 'z'} (Å)")
            if i >= 4:
                ax.set_xlabel("x along C=C (Å)")
        bar = fig.colorbar(im, ax=axes, shrink=.86, pad=.025, aspect=35, extend="min")
        bar.set_label(r"Total electron density $\rho$ ($e\,a_0^{-3}$); logarithmic scale", labelpad=10)
        ticks = [10.**p for p in range(-4, int(math.log10(vmax))+1)]
        bar.set_ticks(ticks)
        plane_label = "Molecular plane: xy, z = 0" if plane == "xy" else "Perpendicular plane through C=C: xz, y = 0"
        fig.suptitle("Ethylene · " + plane_label, fontsize=15)
        for suffix in ("png", "pdf"):
            name = f"figures/ethylene-density-{plane}.{suffix}"
            fig.savefig(OUT / name, dpi=220)
            outputs.append(name)
        plt.close(fig)
        # Individual panels retain the same limits, contour levels and color normalization.
        for key, label in zip(KEYS, LABELS):
            fig, ax = plt.subplots(figsize=(5.8, 5.2), layout="constrained")
            im = panel(ax, key, plane, f"{label} · {records[key]['nao']} AO · {plane} plane")
            ax.set_xlabel("x along C=C (Å)")
            ax.set_ylabel(f"{'y' if plane == 'xy' else 'z'} (Å)")
            bar = fig.colorbar(im, ax=ax, shrink=.86, pad=.025, extend="min")
            bar.set_label(r"$\rho$ ($e\,a_0^{-3}$)")
            for suffix in ("png", "pdf"):
                name = f"figures/{plane}-{key}.{suffix}"
                fig.savefig(OUT / name, dpi=200)
                outputs.append(name)
            plt.close(fig)
    metadata = dict(planes={"xy": "z=0", "xz": "y=0; C=C axis included"},
        shared_color_scale="logarithmic positive total density", vmin_electrons_per_bohr3=vmin,
        vmax_electrons_per_bohr3=vmax, actual_maximum_electrons_per_bohr3=maximum,
        contour_levels_electrons_per_bohr3=levels, interpolation="nearest; no density smoothing",
        files=outputs, note="Only nuclei lying in the chosen plane are marked; H nuclei are outside xz.")
    (OUT / "plot-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
