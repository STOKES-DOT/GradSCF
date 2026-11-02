"""Carbon C0 s/p/d radial response from archived ethane bases; no new SCF.

Run: /opt/anaconda3/bin/python artifacts/nnao-ethane-radial-spd-20261027/analyze.py
After export to NNAO_paper, run this same file against its bundled inputs.
"""
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad

OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nnao-spd-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = {"MACE": "MACE-update3351", "Free": "free-contraction-best"}
COLORS = {"MACE": "#CD732B", "Free": "#246694"}
PRIMITIVE_COLORS = ["#246694", "#C38B1B", "#CB602B", "#71813A", "#AE5B80", "#765F99"]
RADIUS = np.linspace(0, 12, 2401)
BOHR_ANGSTROM = 0.529177210903


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_inputs():
    """Preserve only evidence needed for portable reanalysis, without caches."""
    manifest_path = OUT / "input-manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        for row in manifest:
            assert sha(OUT / row["file"]) == row["sha256"], row["file"]
        return manifest
    root = OUT.parents[1]
    files = []
    torsion = root / "artifacts/nnao-ethane-torsion-20261018"
    stretch = root / "artifacts/nnao-ethane-dissociation-benchmark-20261022"
    final = root / "artifacts/nnao-ethane-dissociation-repair-20261022/final-bases"
    files.append((torsion / "scan-summary.json", "inputs/torsion/scan-summary.json"))
    for angle in range(0, 121, 5):
        for name in ("frozen-bases.jsonl", "structure.json", "configuration.json"):
            relative = f"angle-{angle:03d}/{name}"
            files.append((torsion / relative, "inputs/torsion/" + relative))
    for name in ("nnao-results.json", "structures.json"):
        files.append((stretch / name, "inputs/stretch/" + name))
    for i in range(56):
        relative = f"distance-{i:03d}/frozen-bases.jsonl"
        files.append((final / relative, "inputs/stretch/" + relative))
    manifest = []
    for source, target in files:
        destination = OUT / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert sha(source) == sha(destination)
        manifest.append(dict(file=target, original_source=str(source.relative_to(root)),
                             sha256=sha(source)))
    write_json(manifest_path, manifest)
    return manifest


def matrix(l, alpha, order=0):
    p = l + 1.5 + order / 2
    return math.gamma(p) / (2 * (alpha[:, None] + alpha[None, :]) ** p)


def norm(l, alpha, weights):
    return float(np.sqrt(max(0.0, weights @ matrix(l, alpha) @ weights)))


def radial(l, alpha, weights, radius):
    radius = np.asarray(radius)
    return radius ** l * (np.exp(-radius[..., None] ** 2 * alpha) @ weights)


def shell_data(shell):
    l = int(shell[0])
    rows = np.asarray(shell[1:], dtype=np.float64)
    assert rows.ndim == 2 and rows.shape[1] == 2
    order = np.argsort(-rows[:, 0])
    alpha, raw = rows[order].T
    assert np.all(alpha > 0) and np.all(np.diff(alpha) < 0)
    assert abs(np.linalg.norm(raw) - 1) < 1e-12
    p = l + 1.5
    primitive_norm = np.sqrt(2 * (2 * alpha) ** p / math.gamma(p))
    weights = raw * primitive_norm
    weights /= norm(l, alpha, weights)
    return alpha, raw, weights, primitive_norm, order


def load_scans():
    torsion = read(OUT / "inputs/torsion/scan-summary.json")
    stretch = read(OUT / "inputs/stretch/nnao-results.json")
    geometries = read(OUT / "inputs/stretch/structures.json")
    assert torsion["status"] == "complete" and len(torsion["points"]) == 25
    assert len(stretch["points"]) == len(geometries) == 56
    scans = {"torsion": [], "stretch": []}
    for point in sorted(torsion["points"], key=lambda r: r["angle_deg"]):
        angle = point["angle_deg"]
        folder = OUT / f"inputs/torsion/angle-{int(angle):03d}"
        configuration = read(folder / "configuration.json")
        assert configuration["fixed_core"] and configuration["fixed_exponents"]
        assert configuration["checkpoint"]["update_step"] == 3351
        assert point["coefficient_gradient_converged"]
        scans["torsion"].append((angle, point, folder / "frozen-bases.jsonl",
                                 read(folder / "structure.json")))
    for i, point in enumerate(stretch["points"]):
        assert point["index"] == i and point["status"] == "passed"
        assert point["free_gradient_converged"]
        for key in ("mace_scf", "free_scf"):
            assert point[key]["converged"] and point[key]["stable"]
        assert abs(point["distance_angstrom"] - geometries[i]["distance_angstrom"]) < 1e-12
        path = OUT / f"inputs/stretch/distance-{i:03d}/frozen-bases.jsonl"
        scans["stretch"].append((point["distance_angstrom"], point, path, geometries[i]))
    return scans


def analyze(scans):
    data = {}
    coefficients, metrics, summaries = [], [], []
    fixed_alpha, phase_reference, core = {}, {}, None
    errors = dict(normalization=0.0, independent_quadrature=0.0,
                  frozen_basis_energy=0.0, periodic_closure_l2=0.0)
    for scan, points in scans.items():
        data[scan] = {m: {l: [] for l in range(3)} for m in METHODS}
        for i, (x, point, path, geometry) in enumerate(points):
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert geometry["charge"] == geometry["spin"] == 0
            assert geometry["symbols"][:2] == ["C", "C"]
            for method, label in METHODS.items():
                record = next(r for r in records if r["basis"] == label)
                assert record["nao"] == 44 and record["atom_labels"][0] == "C0"
                np.testing.assert_allclose(record["coords_angstrom"], geometry["coords_angstrom"],
                                           atol=1e-13, rtol=0)
                shells = record["atom_shells"][0]
                assert [s[0] for s in shells] == [0, 0, 1, 2]
                if core is None:
                    core = shells[0]
                assert shells[0] == core
                energy = point["mace_energy_hartree" if method == "MACE" else "free_energy_hartree"]
                error = abs(record["native_energy_hartree"] - energy)
                errors["frozen_basis_energy"] = max(errors["frozen_basis_energy"], error)
                assert error < 1e-9
                for l, shell_index in [(0, 1), (1, 2), (2, 3)]:
                    alpha, raw, weights, pn, order = shell_data(shells[shell_index])
                    assert len(alpha) == (6 if l < 2 else 3)
                    if l not in fixed_alpha:
                        fixed_alpha[l] = alpha.copy()
                        u = RADIUS * radial(l, alpha, weights, RADIUS)
                        reference_sign = 1 if u[np.argmax(abs(u))] > 0 else -1
                        phase_reference[l] = weights.copy() * reference_sign
                    np.testing.assert_array_equal(alpha, fixed_alpha[l])
                    phase = 1 if weights @ matrix(l, alpha) @ phase_reference[l] >= 0 else -1
                    weights = weights * phase
                    rfun = radial(l, alpha, weights, RADIUS)
                    mean = float(weights @ matrix(l, alpha, 1) @ weights)
                    rms = float(np.sqrt(weights @ matrix(l, alpha, 2) @ weights))
                    norm2 = float(weights @ matrix(l, alpha) @ weights)
                    qnorm = quad(lambda r: r ** 2 * float(radial(l, alpha, weights, r)) ** 2,
                                 0, np.inf, epsabs=1e-11, epsrel=1e-11)[0]
                    qmean = quad(lambda r: r ** 3 * float(radial(l, alpha, weights, r)) ** 2,
                                 0, np.inf, epsabs=1e-11, epsrel=1e-11)[0]
                    errors["normalization"] = max(errors["normalization"], abs(norm2 - 1))
                    errors["independent_quadrature"] = max(
                        errors["independent_quadrature"], abs(qnorm - norm2), abs(qmean - mean))
                    series = data[scan][method][l]
                    first = series[0] if series else dict(raw=raw, weights=weights)
                    delta_raw = raw - first["raw"]
                    series.append(dict(x=x, raw=raw, weights=weights, radial=rfun, mean=mean))
                    for j, a in enumerate(alpha):
                        coefficients.append(dict(scan=scan, point_index=i, coordinate=x,
                            coordinate_unit="degree" if scan == "torsion" else "angstrom",
                            atom_index=0, atom_label="C0", channel="spd"[l], primitive=j + 1,
                            source_primitive_zero_based=int(order[j]), exponent_bohr_minus2=float(a),
                            method=method, raw_coefficient=float(raw[j]),
                            raw_change_from_scan_start=float(delta_raw[j]),
                            normalized_primitive_coefficient=float(weights[j] / pn[j]),
                            normalized_radial_weight=float(weights[j]), phase_sign=phase))
                    metrics.append(dict(scan=scan, point_index=i, coordinate=x, method=method,
                        channel="spd"[l], mean_radius_bohr=mean, mean_radius_angstrom=mean * BOHR_ANGSTROM,
                        rms_radius_bohr=rms, radial_l2_from_scan_start=norm(l, alpha, weights - first["weights"]),
                        overlap_with_scan_start=float(weights @ matrix(l, alpha) @ first["weights"]),
                        radial_l2_mace_free=0.0, energy_hartree=energy,
                        normalization_error=abs(norm2 - 1)))
        for l in range(3):
            for method in METHODS:
                series = data[scan][method][l]
                means = np.array([r["mean"] for r in series])
                raw = np.array([r["raw"] for r in series])
                if scan == "torsion":
                    closure = norm(l, fixed_alpha[l], series[-1]["weights"] - series[0]["weights"])
                    errors["periodic_closure_l2"] = max(errors["periodic_closure_l2"], closure)
                    assert closure < 5e-6
                summaries.append(dict(scan=scan, channel="spd"[l], method=method,
                    primitive_count=len(fixed_alpha[l]), points=len(series),
                    mean_radius_initial_bohr=float(means[0]), mean_radius_final_bohr=float(means[-1]),
                    mean_radius_min_bohr=float(min(means)), mean_radius_max_bohr=float(max(means)),
                    mean_radius_peak_to_peak_bohr=float(np.ptp(means)),
                    mean_radius_endpoint_change_percent=float((means[-1] / means[0] - 1) * 100),
                    max_raw_coefficient_change=float(np.max(abs(raw - raw[0])))))
            for i in range(len(points)):
                distance = norm(l, fixed_alpha[l], data[scan]["MACE"][l][i]["weights"] -
                                data[scan]["Free"][l][i]["weights"])
                for row in metrics:
                    if row["scan"] == scan and row["channel"] == "spd"[l] and row["point_index"] == i:
                        row["radial_l2_mace_free"] = distance
    assert errors["normalization"] < 1e-12
    assert errors["independent_quadrature"] < 1e-10
    # Both scans start with the same geometry and the same two basis sets.
    for method in METHODS:
        for l in range(3):
            np.testing.assert_allclose(data["torsion"][method][l][0]["weights"],
                                       data["stretch"][method][l][0]["weights"], atol=1e-12, rtol=0)
    return data, fixed_alpha, coefficients, metrics, summaries, errors


def plot(data, exponents):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "pdf.fonttype": 42, "axes.linewidth": 0.8, "axes.labelcolor": "#242930",
        "text.color": "#242930", "xtick.direction": "in", "ytick.direction": "in"})
    for l, channel in enumerate("spd"):
        fig, axes = plt.subplots(4, 2, figsize=(12, 14.2), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.12, w_pad=0.12, hspace=0.05, wspace=0.04)
        role = "valence" if l < 2 else "polarization"
        fig.suptitle(f"Ethane: carbon C0 {channel} {role} radial response", fontsize=16)
        for col, scan in enumerate(("torsion", "stretch")):
            x = np.array([r["x"] for r in data[scan]["MACE"][l]])
            targets = [0, 30, 60] if scan == "torsion" else [x[0], 3, 8]
            indices = [int(np.argmin(abs(x - t))) for t in targets]
            styles = {"MACE": "-", "Free": "--"}
            snapshot_colors = ["#65717C", "#C38B1B", "#765F99"]
            for method in METHODS:
                for index, color in zip(indices, snapshot_colors):
                    label = f"{x[index]:.0f}°" if scan == "torsion" else f"{x[index]:.3f} Å"
                    if method != "MACE":
                        label = "_nolegend_"
                    y = RADIUS * data[scan][method][l][index]["radial"]
                    axes[0, col].plot(RADIUS, y, color=color, ls=styles[method], lw=1.7, label=label)
            axes[0, col].set_title("Torsion · 25 geometries" if scan == "torsion" else "C–C stretch · 56 geometries")
            axes[0, col].set_xlim(0, 6)
            axes[0, col].set_xlabel(r"Radius from C0 ($a_0$)")
            axes[0, col].set_ylabel(r"$u_l(r)=rR_l(r)$ ($a_0^{-1/2}$)")
            handles, labels = axes[0, col].get_legend_handles_labels()
            handles += [Line2D([], [], color="#333333", ls=styles[m], label=m) for m in METHODS]
            axes[0, col].legend(handles=handles, frameon=False, ncol=3, fontsize=9, loc="upper right")
            axes[0, col].axhline(0, color="#AAB0B5", lw=0.6)
            dc_arrays = []
            for row, method in [(1, "MACE"), (2, "Free")]:
                raw = np.array([p["raw"] for p in data[scan][method][l]])
                dc = raw - raw[0]
                dc_arrays.append(dc)
                for j, alpha in enumerate(exponents[l]):
                    axes[row, col].plot(x, dc[:, j], color=PRIMITIVE_COLORS[j], lw=1.6,
                                        marker=["o", "s", "^", "D", "v", "P"][j],
                                        markersize=2.6, markevery=2 if scan == "torsion" else 4,
                                        label=rf"$c_{{{j+1}}}$: {alpha:.4g}")
                axes[row, col].set_title(f"{method}: every primitive coefficient", fontsize=11)
                axes[row, col].set_ylabel(r"$c_i(q)-c_i(q_0)$")
                axes[row, col].axhline(0, color="#AAB0B5", lw=0.6)
                axes[row, col].legend(frameon=False, ncol=3, fontsize=8.7, loc="upper center",
                                      title=r"Primitive exponent ($a_0^{-2}$)", title_fontsize=9)
                axes[row, col].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
            # Same coefficient scale for MACE/free within each process.
            low = min(float(a.min()) for a in dc_arrays)
            high = max(float(a.max()) for a in dc_arrays)
            pad = max(0.0001, (high - low) * 0.10)
            legend_space = max(0.0002, (high - low) * (0.50 if l < 2 else 0.38))
            for row in (1, 2):
                axes[row, col].set_ylim(low - pad, high + legend_space)
            for method in METHODS:
                means = np.array([p["mean"] for p in data[scan][method][l]])
                axes[3, col].plot(x, means, color=COLORS[method], lw=1.9, ls=styles[method],
                                  marker="s" if method == "MACE" else "o", markersize=2.7,
                                  label=method)
            axes[3, col].set_ylabel(r"Mean radius $\langle r\rangle$ ($a_0$)")
            axes[3, col].ticklabel_format(axis="y", useOffset=False)
            axes[3, col].legend(frameon=False, fontsize=9)
            for row in (1, 2, 3):
                axes[row, col].set_xlim(x[0], x[-1])
                axes[row, col].set_xlabel("Methyl rotation (degree)" if scan == "torsion" else "C–C distance (Å)")
                if scan == "torsion":
                    axes[row, col].set_xticks([0, 30, 60, 90, 120])
            for row in range(4):
                axes[row, col].grid(color="#E1E4E8", lw=0.55, alpha=0.7)
                axes[row, col].set_axisbelow(True)
                axes[row, col].text(0.025, 0.96, f"({chr(97 + row * 2 + col)})",
                                    transform=axes[row, col].transAxes, va="top", fontsize=10)
        common_min = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
        common_max = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
        for ax in axes[0]:
            ax.set_ylim(common_min, common_max)
        for suffix in ("png", "pdf"):
            fig.savefig(OUT / f"figures/carbon-{channel}-radial-coefficients.{suffix}", dpi=190)
        plt.close(fig)


def main():
    start = time.perf_counter()
    manifest = freeze_inputs()
    for name in ("figures", "tables", "arrays"):
        (OUT / name).mkdir(exist_ok=True)
    scans = load_scans()
    data, exponents, coefficients, metrics, summaries, errors = analyze(scans)
    write_csv(OUT / "tables/primitive-coefficients.csv", coefficients)
    write_csv(OUT / "tables/radial-metrics.csv", metrics)
    write_csv(OUT / "tables/radial-summary.csv", summaries)
    ranges = []
    for scan in data:
        for l, channel in enumerate("spd"):
            for method in METHODS:
                rows = data[scan][method][l]
                c = np.array([r["raw"] for r in rows])
                for i, a in enumerate(exponents[l]):
                    ranges.append(dict(scan=scan, channel=channel, method=method, primitive=i+1,
                        exponent_bohr_minus2=float(a), initial_coefficient=float(c[0, i]),
                        final_coefficient=float(c[-1, i]), minimum_coefficient=float(min(c[:, i])),
                        maximum_coefficient=float(max(c[:, i])),
                        max_absolute_change=float(max(abs(c[:, i] - c[0, i])))))
    write_csv(OUT / "tables/coefficient-ranges.csv", ranges)
    arrays = {"radius_bohr": RADIUS}
    for scan in data:
        arrays[f"{scan}_coordinate"] = np.array([r["x"] for r in data[scan]["MACE"][0]])
        for method in METHODS:
            for l, channel in enumerate("spd"):
                prefix = f"{scan}_{method.lower()}_{channel}"
                arrays[f"{prefix}_R"] = np.array([r["radial"] for r in data[scan][method][l]])
                arrays[f"{prefix}_raw_coefficients"] = np.array([r["raw"] for r in data[scan][method][l]])
                arrays[f"{prefix}_normalized_weights"] = np.array([r["weights"] for r in data[scan][method][l]])
                arrays[f"{channel}_exponents_bohr_minus2"] = exponents[l]
    np.savez_compressed(OUT / "arrays/radial-functions.npz", **arrays)
    plot(data, exponents)
    verification = dict(status="passed", geometries=81, frozen_basis_records=162, normalized_functions=len(metrics),
        coefficient_rows=len(coefficients), source_files=len(manifest), errors=errors,
        fixed_core=True, fixed_exponents=True, source_checkpoint_update=3351,
        source_method="all-electron DF-RHF / def2-universal-jkfit; spherical; float64; 44 AOs",
        new_scf_calls=0, runtime_python=platform.python_version(), runtime_platform=platform.platform(),
        elapsed_seconds=time.perf_counter() - start)
    write_json(OUT / "verification.json", verification)
    print(json.dumps(dict(verification=verification, summary=summaries), indent=2))


if __name__ == "__main__":
    main()
