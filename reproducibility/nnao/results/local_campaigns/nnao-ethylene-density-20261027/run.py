"""Six matched DF-RHF ethylene density maps, with native implicit free contraction.

Run from GradSCF with its pinned mace-jax on PYTHONPATH. Native integrals and
SCF are GradSCF; PySCF is used only for an independent frozen-basis comparison.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("GRADSCF_NATIVE_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from gradscf.data.molecule import ANGSTROM_TO_BOHR
from gradscf.integrals._native import register_integrals
from gradscf.integrals.basis.data import load_basis_from_snapshot
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, file_hash, scientific_source_hash
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
from gradscf.scf.stability import stabilize_df_rhf
from gradscf.integrals.molecular.density_fitting import unpack_factors
from density import evaluate_ao, evaluate_density

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from comparisons.nnao_core.native import make_inputs

MODEL = Path("/Users/jiaoyuan/Documents/GitHub/NNAO_paper/data/computational_examples/2026-10-21/models/update-3351")
NAMES = ["NNAO / MACE", "NNAO / free contraction", "STO-6G", "STO-6G+P", "cc-pVTZ", "cc-pVDZ"]
KEYS = ["mace", "free", "sto6g", "sto6g-p", "cc-pvtz", "cc-pvdz"]


def progress(phase, **fields):
    atomic_json(OUT / "progress.json", dict(phase=phase, **fields))
    print(phase, fields, flush=True)


def normalize(c, layout):
    c = np.array(c, copy=True)
    for atom, slot, count, columns in zip(layout.shell_atoms, layout.slots,
            layout.topology.primitive_counts, layout.topology.contraction_counts):
        if slot >= 0:
            for column in range(slot, slot + columns):
                c[atom, column, :count] /= np.linalg.norm(c[atom, column, :count])
    return c


def coefficient_metric(layout, mask):
    # Same invertible primitive-overlap whitening as the corrected ethane scan.
    positions = np.flatnonzero(mask.ravel())
    transform = np.eye(len(positions))
    for atom, slot, count, columns, ell, alpha in zip(layout.shell_atoms, layout.slots,
            layout.topology.primitive_counts, layout.topology.contraction_counts,
            layout.topology.angular_momenta, layout.parameters.exponents):
        if slot < 0:
            continue
        alpha = np.asarray(alpha)
        overlap = (2 * np.sqrt(alpha[:, None] * alpha[None, :]) /
                   (alpha[:, None] + alpha[None, :])) ** (ell + 1.5)
        block = np.linalg.solve(np.linalg.cholesky(overlap).T, np.eye(count))
        np.testing.assert_allclose(block.T @ overlap @ block, np.eye(count), atol=1e-10, rtol=0)
        for column in range(slot, slot + columns):
            flat = np.ravel_multi_index((np.full(count, atom), np.full(count, column), np.arange(count)), mask.shape)
            indices = np.searchsorted(positions, flat)
            transform[np.ix_(indices, indices)] = block
    return transform


def optimize(experiment, layout, initial, mask, name):
    template = normalize(initial, layout)
    transform = coefficient_metric(layout, mask)
    first = np.linalg.solve(transform, template[mask])
    current, history = {}, []
    start = time.perf_counter()

    def unpack(x):
        c = template.copy()
        c[mask] = transform @ x
        return c

    def fun(x):
        if "x" not in current or not np.array_equal(x, current["x"]):
            energy, g, info = experiment.evaluate_device(jnp.asarray(unpack(x)))
            g = np.asarray(g)
            current.update(x=np.array(x), energy=energy, g=g, info=info)
        return current["energy"], transform.T @ current["g"][mask]

    def callback(x):
        energy, _ = fun(x)
        g, c = current["g"].copy(), unpack(x)
        for atom, slot, count, columns in zip(layout.shell_atoms, layout.slots,
                layout.topology.primitive_counts, layout.topology.contraction_counts):
            if slot >= 0:
                for column in range(slot, slot + columns):
                    g[atom, column, :count] *= np.linalg.norm(c[atom, column, :count])
        norm = float(np.linalg.norm(g[mask]))
        history.append(dict(iteration=len(history) + 1, energy_hartree=energy, gradient_norm=norm))
        if len(history) % 20 == 0:
            atomic_json(OUT / f"optimization/{name}-history.json", history)
            progress("free_contraction", start=name, **history[-1])
        if norm <= 3e-6:
            raise StopIteration

    result = minimize(fun, first, method="L-BFGS-B", jac=True, callback=callback,
        options=dict(maxiter=1000, maxfun=4000, maxls=40, maxcor=20, ftol=1e-15, gtol=1e-8))
    c = normalize(unpack(result.x), layout)
    energy, gradient, info = experiment.evaluate_device(jnp.asarray(c))
    norm = float(np.linalg.norm(np.asarray(gradient)[mask]))
    report = dict(start=name, energy_hartree=energy, gradient_norm=norm,
        accepted=norm <= 1e-5, iterations=int(result.nit), evaluations=int(result.nfev),
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        seconds=time.perf_counter() - start, scf=info)
    atomic_json(OUT / f"optimization/{name}-history.json", history)
    atomic_json(OUT / f"optimization/{name}-result.json", report)
    np.savez(OUT / f"optimization/{name}-final.npz", coefficients=c, gradient=np.asarray(gradient))
    progress("free_start_complete", **report)
    return report, c


def stable_state(data, nelectron):
    n = len(data["overlap"])
    kwargs = dict(overlap=jnp.asarray(data["overlap"]), hcore=jnp.asarray(data["hcore"]),
        eri=None, df_factors=unpack_factors(jnp.asarray(data["factors"]), n), nelectron=nelectron,
        nuclear_repulsion=jnp.asarray(data["enuc"]), ao=jnp.zeros((0, n)),
        ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0))
    config = RKSConfig(xc_spec="hf", jk_backend="df", max_cycle=200,
        conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)
    result = run_rks_from_integrals_traceable(**kwargs, config=config)
    result, stability = stabilize_df_rhf(result, **kwargs, config=config)
    assert bool(result.converged) and bool(stability["stable"])
    s, d, f = np.asarray(result.overlap_matrix), np.asarray(result.density_matrix), np.asarray(result.fock_matrix)
    residual = float(np.linalg.norm(f @ d @ s - s @ d @ f))
    assert residual < 1e-7
    assert abs(np.einsum("ij,ji", d, s) - nelectron) < 1e-8
    return result, dict(converged=True, internal_stable=True, orbital_residual=residual,
                        electron_number=float(np.einsum("ij,ji", d, s)), cycles=int(result.cycles))


def main():
    start = time.perf_counter()
    for folder in ("inputs", "optimization", "states", "arrays", "figures", "cache"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    row = dict(structure_id="GradSCF-example/ethylene_boys", symbols=["C", "C", "H", "H", "H", "H"],
        atomic_numbers=[6, 6, 1, 1, 1, 1], charge=0, spin=0, nelectron=16, unit="Angstrom",
        coords_angstrom=[[-.6695, 0, 0], [.6695, 0, 0], [-1.2321, -.9289, 0],
                         [-1.2321, .9289, 0], [1.2321, -.9289, 0], [1.2321, .9289, 0]],
        geometry_source="examples/localization/ethylene_boys.py", geometry_optimized=False)
    atomic_json(OUT / "inputs/structure.json", row)
    (OUT / "inputs/ethylene.xyz").write_text("6\nEthylene; charge=0; 2S=0; Angstrom; fixed geometry\n" +
        "".join(f"{s} {x:.12f} {y:.12f} {z:.12f}\n" for s, (x, y, z) in zip(row["symbols"], row["coords_angstrom"])))
    snapshot = json.loads((MODEL / "inference-snapshot.json").read_text())
    for name in ("model-config.json", "fixed-candidate.json", "inference-snapshot.json", "parameter-export.json"):
        shutil.copy2(MODEL / name, OUT / "inputs" / name)
    configuration = dict(method="all-electron DF-RHF", auxbasis="def2-universal-jkfit", cart=False,
        dtype="float64", angstrom_to_bohr=ANGSTROM_TO_BOHR, geometry=row, model=snapshot, fixed_core=True, fixed_exponents=True,
        free_optimizer="two-start L-BFGS-B; primitive-overlap whitening; existing implicit SCF gradient",
        free_gradient_tolerance=1e-5, finite_differences=False,
        sto6g_p=dict(carbon_d_exponent_bohr_minus2=.8, hydrogen_p_exponent_bohr_minus2=1.1),
        source_hash=scientific_source_hash(ROOT), native_sha256=file_hash(register_integrals()._name),
        python=platform.python_version(), platform=platform.platform(), jax=jax.__version__,
        backend=jax.default_backend())
    atomic_json(OUT / "configuration.json", configuration)
    candidate = json.loads((OUT / "inputs/fixed-candidate.json").read_text())
    atom = list(zip(row["symbols"], row["coords_angstrom"]))
    layout = bind_pool_candidate(prepare_direct_basis(atom, basis_family=snapshot["basis_family"],
        core_primitives=snapshot["core_primitives"], cart=False), candidate)
    assert layout.topology.nao == 36
    coefficient_file = OUT / "inputs/mace-coefficients.npz"
    if not coefficient_file.exists():
        progress("mace_initialization")
        initial, predict, _ = make_mace_predictor(snapshot["seed"], snapshot["model_config"])
        with np.load(MODEL / "parameters.npz") as saved:
            vector = np.array(saved["parameters"])
        assert vector.shape == initial.shape
        assert hashlib.sha256(vector.tobytes()).hexdigest() == snapshot["parameter_sha256"]
        # Check the current model/tree against the archived ethane prediction.
        source = ROOT / "artifacts/nnao-ethane-torsion-20261018/angle-000/frozen-bases.jsonl"
        ethane = next(r for r in map(json.loads, source.read_text().splitlines()) if r["basis"] == "MACE-update3351")
        graph = build_graph([6, 6] + [1] * 6, ethane["coords_angstrom"],
            element_order=snapshot["model_config"]["elements"], cutoff=snapshot["model_config"]["cutoff"])
        check_layout = bind_pool_candidate(prepare_direct_basis(list(zip(ethane["symbols"], ethane["coords_angstrom"])),
            basis_family=snapshot["basis_family"], core_primitives=snapshot["core_primitives"], cart=False), candidate)
        predicted = check_layout.atom_shells(check_layout.bind(predict(jnp.asarray(vector), graph)))
        for a, b in zip(predicted, ethane["atom_shells"]):
            for sa, sb in zip(a, b):
                assert sa[0] == sb[0]
                np.testing.assert_allclose(sa[1:], sb[1:], atol=1e-11, rtol=1e-11)
        graph = build_graph(row["atomic_numbers"], row["coords_angstrom"],
            element_order=snapshot["model_config"]["elements"], cutoff=snapshot["model_config"]["cutoff"])
        mace = normalize(np.asarray(predict(jnp.asarray(vector), graph)), layout)
        np.savez(coefficient_file, coefficients=mace)
        atomic_json(OUT / "model-lineage-check.json", dict(status="passed", checkpoint_update=3351,
            archived_ethane_basis_reproduced=True, tolerance=1e-11, parameter_sha256=snapshot["parameter_sha256"]))
        del predict, vector, initial
        jax.clear_caches()
    mace = np.load(coefficient_file)["coefficients"]
    mask = np.zeros_like(mace, dtype=bool)
    for atom_index, slot, count, columns in zip(layout.shell_atoms, layout.slots,
            layout.topology.primitive_counts, layout.topology.contraction_counts):
        if slot >= 0:
            mask[atom_index, slot:slot+columns, :count] = True
    progress("native_fixed_pool")
    experiment, manifest = prepare_fixed_pool(row, layout, OUT / "cache/pool.npz", storage="device",
        auxbasis=configuration["auxbasis"], implicit_tolerance=1e-8)
    atomic_json(OUT / "pool-summary.json", {k: v for k, v in manifest.items() if k != "identity"})
    solutions = []
    for name, initial in (("mace", mace), ("reference", np.asarray(layout.reference_outputs()))):
        report_file = OUT / f"optimization/{name}-result.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            c = np.load(OUT / f"optimization/{name}-final.npz")["coefficients"]
        else:
            report, c = optimize(experiment, layout, initial, mask, name)
        if report["accepted"]:
            solutions.append((report["energy_hartree"], c, name))
    assert len(solutions) == 2, "Both starts must meet the gradient criterion before reporting free contraction."
    energy, free, selected = min(solutions, key=lambda r: r[0])
    atomic_json(OUT / "optimization/selection.json", dict(selected_start=selected, energy_hartree=energy,
        endpoint_spread_hartree=max(s[0] for s in solutions)-min(s[0] for s in solutions)))
    basis_sets = [layout.atom_shells(layout.bind(jnp.asarray(c))) for c in (mace, free)]
    for name in ("sto-6g", "sto-6g+p", "cc-pvtz", "cc-pvdz"):
        shells = [load_basis_from_snapshot("sto-6g" if name == "sto-6g+p" else name, s) for s in row["symbols"]]
        if name == "sto-6g+p":
            shells = [b + [[2 if s == "C" else 1, [.8 if s == "C" else 1.1, 1.0]]]
                      for s, b in zip(row["symbols"], shells)]
        basis_sets.append(shells)
    frozen = [dict(basis=name, key=key, symbols=row["symbols"], coords_angstrom=row["coords_angstrom"],
                   atom_labels=[f"{s}{i}" for i, s in enumerate(row["symbols"])], atom_shells=basis)
              for name, key, basis in zip(NAMES, KEYS, basis_sets)]
    atomic_json(OUT / "inputs/frozen-bases.json", frozen)
    del experiment
    jax.clear_caches()
    # A common molecular plane and identical pixel positions for all six maps.
    x = np.linspace(-2.6, 2.6, 641)
    y = np.linspace(-2.2, 2.2, 561)
    xx, yy = np.meshgrid(x, y)
    planes = {
        "xy": np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1) * ANGSTROM_TO_BOHR,
        "xz": np.stack([xx.ravel(), np.zeros(xx.size), yy.ravel()], axis=1) * ANGSTROM_TO_BOHR,
    }
    centers = np.asarray(row["coords_angstrom"]) * ANGSTROM_TO_BOHR
    arrays = dict(x_angstrom=x, vertical_angstrom=y, xy_z_angstrom=np.array(0.), xz_y_angstrom=np.array(0.),
                  angstrom_to_bohr=np.array(ANGSTROM_TO_BOHR))
    records = []
    # Independent comparison is deliberately kept in this experiment harness.
    from pyscf import gto, scf, lib
    lib.num_threads(1)
    rng = np.random.default_rng(0)
    probe = np.vstack((centers, rng.uniform(-4, 4, (200, 3))))
    for entry in frozen:
        key, basis = entry["key"], entry["atom_shells"]
        tick = time.perf_counter()
        state_file, record_file = OUT / f"states/{key}.npz", OUT / f"states/{key}.json"
        progress("density_and_reference", basis=entry["basis"])
        if state_file.exists() and record_file.exists():
            state = dict(np.load(state_file))
            report = json.loads(record_file.read_text())
        else:
            data = make_inputs(row, basis, auxbasis=configuration["auxbasis"])
            result, report = stable_state(data, row["nelectron"])
            state = dict(density=np.asarray(result.density_matrix), mo=np.asarray(result.mo_coeff),
                occupations=np.asarray(result.mo_occ), overlap=np.asarray(result.overlap_matrix))
            report.update(basis=entry["basis"], key=key, nao=len(state["density"]),
                          energy_hartree=float(result.total_energy))
            if key == "free":
                assert abs(report["energy_hartree"] - energy) < 1e-9
            np.savez(state_file, **state)
            # Test-only PySCF: same individual atom shells and auxiliary basis.
            labels = entry["atom_labels"]
            mol = gto.M(atom=list(zip(labels, row["coords_angstrom"])),
                basis=dict(zip(labels, basis)), unit="Angstrom", cart=False, verbose=0)
            reference = scf.RHF(mol).density_fit(auxbasis=configuration["auxbasis"])
            reference.conv_tol = 1e-12
            reference.conv_tol_grad = 1e-9
            reference.max_cycle = 200
            reference.kernel()
            assert reference.converged
            energy_error = abs(float(reference.e_tot) - report["energy_hartree"])
            assert energy_error < 1e-8, energy_error
            analytic = evaluate_ao(probe, centers, basis)
            pyscf_ao = mol.eval_gto("GTOval_sph", probe)
            ao_error = float(np.max(abs(analytic - pyscf_ao)))
            assert ao_error < 1e-10, ao_error
            rho_a = np.einsum("pi,ij,pj->p", analytic, state["density"], analytic)
            rho_b = np.einsum("pi,ij,pj->p", pyscf_ao, state["density"], pyscf_ao)
            report.update(pyscf_energy_hartree=float(reference.e_tot), pyscf_energy_error_hartree=energy_error,
                ao_probe_max_error=ao_error, density_probe_max_error=float(np.max(abs(rho_a - rho_b))),
                frozen_basis_density_matrix_max_error=float(np.max(abs(reference.make_rdm1() - state["density"]))),
                pyscf_version=__import__("pyscf").__version__, probe_points=len(probe))
            atomic_json(record_file, report)
        for plane_name, plane in planes.items():
            rho = evaluate_density(plane, centers, basis, state["mo"], state["occupations"])
            arrays[f"{plane_name}_{key}"] = rho.reshape(xx.shape)
            report[f"{plane_name}_minimum_density"] = float(rho.min())
            report[f"{plane_name}_maximum_density"] = float(rho.max())
        report["seconds_with_plane_evaluation"] = time.perf_counter() - tick
        records.append(report)
        atomic_json(OUT / "partial-results.json", records)
        jax.clear_caches()
    np.savez_compressed(OUT / "arrays/plane-densities.npz", **arrays)
    for report in records:
        for plane_name in planes:
            reference = arrays[f"{plane_name}_cc-pvtz"]
            delta = arrays[f"{plane_name}_{report['key']}"] - reference
            report[f"{plane_name}_rms_density_error_vs_ccpvtz"] = float(np.sqrt(np.mean(delta * delta)))
            report[f"{plane_name}_relative_l1_vs_ccpvtz"] = float(np.sum(abs(delta)) / np.sum(reference))
    atomic_json(OUT / "results.json", dict(status="complete", requested=6, passed=6, records=records,
        elapsed_seconds=time.perf_counter() - start, density_definition="spin-summed total rho, electrons/bohr^3",
        grid=dict(shape=list(xx.shape), planes={"xy": "z=0", "xz": "y=0; contains C=C axis"},
                  x_angstrom=[float(x[0]), float(x[-1])], vertical_angstrom=[float(y[0]), float(y[-1])]),
        finite_differences=False))
    with (OUT / "summary.csv").open("w", newline="") as stream:
        names = ["basis", "nao", "energy_hartree", "electron_number", "orbital_residual",
                 "pyscf_energy_error_hartree", "ao_probe_max_error", "xy_relative_l1_vs_ccpvtz",
                 "xz_relative_l1_vs_ccpvtz"]
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    progress("complete", seconds=time.perf_counter() - start, passed=6)


if __name__ == "__main__":
    main()
