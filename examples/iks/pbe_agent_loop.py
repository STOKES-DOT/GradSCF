"""Replay the two-round PBE functional-design pilot from its verified bundle.

Set GRADSCF_IKS_BUNDLE to the directory containing BUNDLE.json. The example
uses cached fixed-reference E/F tables and invokes no external model or SCF.
All fitting, energy/potential AD and response algorithms live in src/.
"""
from pathlib import Path
import csv
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gradscf_tools.pbe_loop import load_affine_cache
from pbe_agent_functionals import get_candidate


bundle = Path(os.environ.get("GRADSCF_IKS_BUNDLE", "pbe_loop_results"))
metadata = json.loads((bundle / "BUNDLE.json").read_text())
original_root = Path(metadata["original_remote_root"])
run = bundle / metadata["run_relative"]
winner = json.loads((run / "frozen_winner.json").read_text())
test = json.loads((bundle / metadata["test_relative"] / "test_report.json").read_text())

# Preserve the exact JSON/NPZ contents and hashes when moving the bundle.
# Only file locations change; the shared loader receives explicit local paths.
def local_path(original_path):
    return bundle / Path(original_path).relative_to(original_root)


tables = json.loads((bundle / metadata["table_manifests"][winner["candidate_id"]]).read_text())
train_caches = []
for row in tables["rows"]:
    if row["split"] != "train":
        continue
    teacher_path = local_path(row["teacher_json"])
    teacher = json.loads(teacher_path.read_text())
    train_caches.append(load_affine_cache(
        local_path(row["cache"]), teacher_json=teacher_path,
        reference_npz=local_path(teacher["reference_artifact"]["path"]),
        expected_sha256=row["sha256"],
    ))

# Show parameter AD on the same scalar loss used in training. Shift the point
# slightly because MAE is nonsmooth when an individual energy error is zero.
# Its point gradient is not an optimality certificate at such a kink.
theta = jnp.asarray(winner["parameters"], dtype=jnp.float64)
if theta.size:
    proposal = get_candidate(winner["candidate_id"])
    print("External scalar-energy terms:", [term.name for term in proposal["terms"]])
    loss = lambda params: jnp.mean(jnp.stack([cache.loss(params) for cache in train_caches]))
    point = theta + 1e-4
    value, gradient = jax.jit(jax.value_and_grad(loss))(point)
    direction = jnp.arange(1, theta.size + 1, dtype=theta.dtype)
    direction /= jnp.linalg.norm(direction)
    step = 1e-7
    finite_difference = (loss(point + step * direction) - loss(point - step * direction)) / (2 * step)
    ad_direction = jnp.dot(gradient, direction)
    np.testing.assert_allclose(ad_direction, finite_difference, rtol=1e-5, atol=1e-6)
    print("Training loss at shifted parameters:", float(value))
    print("AD parameter gradient:", np.asarray(gradient))
    print("Directional AD / finite difference:", float(ad_direction), float(finite_difference))

# The winner was frozen using validation alone. These test results were opened
# once afterward; they are not passed back to the formula-design agent.
ids = ["c000", "c001", "c002"]
records = [json.loads((run / "candidates" / name / "candidate.json").read_text()) for name in ids]
validation_rmse = [1000 * r["scores"]["metrics"]["validation"]["energy_wrmsd_hartree"] for r in records]
validation_rotation = [r["scores"]["metrics"]["validation"]["rotation_norm2_mean"] for r in records]
colors = ["#687887", "#007C91", "#D17A22"]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
axes[0].bar(ids, validation_rmse, color=colors)
axes[0].set_ylabel("Validation energy WRMSD (mHa)")
axes[0].set_yscale("log")
axes[0].set_ylim(min(validation_rmse) * .6, max(validation_rmse) * 2)
for index, value in enumerate(validation_rmse):
    axes[0].text(index, value * 1.08, f"{value:.2f}", ha="center", fontsize=9)
axes[0].set_title("Formula search")
axes[1].bar(ids, validation_rotation, color=colors)
axes[1].set_ylabel("Mean squared orbital-rotation response")
axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
axes[1].set_title("Stationarity diagnostic")

predictions = test["predictions"]
for role, label, color in (("baseline", "PBE", colors[0]), ("winner", winner["candidate_id"], colors[1])):
    errors = [1000 * row["signed_error"] for row in predictions if row["model_role"] == role]
    axes[2].plot(np.arange(1, 6), errors, "o-", label=label, color=color)
axes[2].axhline(0, color="black", linewidth=.7)
axes[2].set_xticks(np.arange(1, 6))
axes[2].set_xlabel("Held-out molecule index")
axes[2].set_ylabel("Signed total-energy error (mHa)")
axes[2].legend(frameon=False)
axes[2].set_title("One final test evaluation")
fig.suptitle("PBE analytic-functional pilot: fixed inverse KS references", fontsize=12)
output = bundle / "figures"
output.mkdir(exist_ok=True)
for extension in ("png", "pdf"):
    fig.savefig(output / f"pbe_agent_loop.{extension}", dpi=300)
plt.close(fig)
print("Frozen winner:", winner["candidate_id"], winner["parameters"])
print("Held-out metrics:", json.dumps(test["metrics"], indent=2))
print("Figures:", output)

# Actual numerical output is documented in README.md after remote execution.
# This demonstrates a fixed-reference design/assessment loop. It is not an
# atomization-energy benchmark or a validation of self-consistent accuracy.
