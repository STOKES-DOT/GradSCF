"""Plot accepted training losses from the read-only remote snapshot."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "loss-snapshot.json").read_text())
rows = data["history"]
epoch = np.array([r["epoch"] for r in rows])
energy = np.array([r["loss_hartree"] for r in rows])
assert np.all(np.diff(epoch) == 1) and epoch[0] == 0
assert np.isfinite(energy).all()
decrease = (energy[0] - energy) * 1000
with (root / "loss-curve.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "mean_energy_Eh", "decrease_from_initial_mEh"])
    writer.writerows(zip(epoch, energy, decrease))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, values, label in zip(axes, [energy, decrease],
        ["Mean SCF energy / molecule (Eh)", "Energy decrease from initialization (mEh / molecule)"]):
    ax.plot(epoch, values, color="#196b91", linewidth=2.3)
    ax.scatter(epoch[-1], values[-1], color="#196b91", s=32, zorder=3)
    ax.axvline(22, color="#bf7933", linestyle="--", linewidth=1.2)
    ax.text(0.98, 0.06 if ax is axes[1] else 0.95, "Dashed line: restart from epoch 22", transform=ax.transAxes,
            ha="right", va="bottom" if ax is axes[1] else "top", color="#986027", fontsize=9)
    ax.set_xlabel("Accepted epoch")
    ax.set_ylabel(label)
    ax.grid(alpha=0.18)
    ax.set_xlim(0, epoch[-1] + max(5, .03*epoch[-1]))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
axes[0].set_title("Training loss (lower is better)", loc="left", fontsize=12)
axes[1].set_title("Cumulative energy improvement", loc="left", fontsize=12)
axes[0].text(0.04, 0.06, f"Epoch {epoch[-1]}: {energy[-1]:.10f} Eh", transform=axes[0].transAxes,
             fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
axes[1].text(0.12, 0.85, f"{decrease[-1]:.4f} mEh / molecule", transform=axes[1].transAxes,
             fontsize=10)
fig.suptitle("NNAO | 200-molecule fixed-exponent MACE training", fontsize=16, y=0.98)
fig.text(0.5, 0.015,
         f"Accepted checkpoints only | Adam lr = 1e-4 | Snapshot: {data['captured_at']} | No validation split",
         ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=(0, 0.055, 1, 0.92))
for ext in ("png", "pdf"):
    fig.savefig(root / f"loss-curve.{ext}", dpi=200)
print(json.dumps({"epoch": int(epoch[-1]), "energy_Eh": float(energy[-1]),
                  "decrease_mEh": float(decrease[-1]),
                  "all_steps_decrease": bool(np.all(np.diff(energy) < 0))}))
