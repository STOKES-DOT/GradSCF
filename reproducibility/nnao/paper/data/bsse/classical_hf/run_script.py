"""Native GradSCF RHF counterpoise examples on three fixed S22 geometries.

Run from the GradSCF checkout with PYTHONPATH=src. This example uses CPU
float64, spherical STO-3G/3-21G/def2-SVP and exact packed AO integrals.
The optional PySCF calculation is an independent validation, not the
producer of GradSCF integrals or SCF energies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import jax
import jax.numpy as jnp
import numpy as np

import gradscf
from gradscf import integrals, scf
from gradscf.data.molecule import MoleculeSpec, atomic_number, ANGSTROM_TO_BOHR

HARTREE_TO_KCAL_MOL = 627.5094740631
CONFIG = scf.RHFConfig(
    max_cycle=200,
    conv_tol=1e-12,
    conv_tol_density=1e-10,
    conv_tol_grad=1e-9,
    orthogonalization_eps=1e-10,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_spec(system, component):
    symbols = tuple(system["symbols"])
    xyz = np.asarray(system["positions_angstrom"], dtype=np.float64)
    charges = np.array([atomic_number(s) for s in symbols], dtype=np.float64)
    fragments = system["fragments"]
    if component in ("A", "B"):
        indices = np.asarray(fragments[0 if component == "A" else 1])
        symbols = tuple(symbols[i] for i in indices)
        xyz, charges = xyz[indices], charges[indices]
    elif component in ("A_ghost_B", "B_ghost_A"):
        ghost = fragments[1 if component == "A_ghost_B" else 0]
        charges[np.asarray(ghost)] = 0.0
    elif component != "AB":
        raise ValueError(component)
    return MoleculeSpec(
        symbols=symbols,
        coords_bohr=xyz * ANGSTROM_TO_BOHR,
        charges=charges,
        charge=0,
        spin=0,
        unit="Angstrom",
    )


def reference_energy(spec, basis_name, overlap, hcore, enuc):
    from pyscf import gto, lib, scf as pyscf_scf

    lib.num_threads(1)
    atom = [
        (sym if z else "ghost-" + sym, tuple(xyz))
        for sym, z, xyz in zip(spec.symbols, spec.charges, spec.coords_bohr)
    ]
    mol = gto.M(atom=atom, unit="Bohr", basis=basis_name, cart=False,
                charge=spec.charge, spin=spec.spin, verbose=0)
    mf = pyscf_scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 200
    energy = float(mf.kernel())
    if not mf.converged or mol.nelectron != spec.nelectron:
        raise RuntimeError("PySCF reference failed convergence/electron checks.")
    checks = {
        "overlap_max_abs": float(np.max(np.abs(
            overlap - mol.intor("int1e_ovlp")))),
        "hcore_max_abs": float(np.max(np.abs(hcore - mf.get_hcore()))),
        "nuclear_repulsion_abs": abs(enuc - float(mol.energy_nuc())),
    }
    if max(checks.values()) > 1e-9:
        raise RuntimeError(f"Native/reference one-electron mismatch: {checks}")
    return energy, checks


def calculate(spec, basis_name, check_pyscf, shared=None):
    started = time.perf_counter()
    reused = shared is not None
    if spec.spin or spec.nelectron % 2:
        raise ValueError("This example requires neutral closed-shell fragments.")
    topology, parameters = integrals.prepare_basis(spec, basis_name, cart=False)
    plan = integrals.make_plan(topology, backend="native")
    if shared is None:
        shared = {
            "overlap": plan.evaluate("overlap", parameters),
            "kinetic": plan.evaluate("kinetic", parameters),
            "eri": plan.evaluate("eri", parameters, aosym="s4"),
            "symbols": spec.symbols,
            "coords": np.asarray(spec.coords_bohr),
        }
    else:
        if shared["symbols"] != spec.symbols:
            raise ValueError("Shared integrals require identical basis centers.")
        np.testing.assert_array_equal(shared["coords"], spec.coords_bohr)
    s, eri = shared["overlap"], shared["eri"]
    h = shared["kinetic"] + plan.evaluate("nuclear", parameters)
    enuc = scf.nuclear_repulsion_energy(
        parameters.nuclear_coords, jnp.asarray(topology.nuclear_charges))
    jax.block_until_ready((s, h, eri, enuc))
    integral_seconds = time.perf_counter() - started
    scf_started = time.perf_counter()
    result = scf.run_rhf_from_integrals(
        overlap=s, hcore=h, eri=eri, nelectron=spec.nelectron,
        nuclear_repulsion=enuc, config=CONFIG,
    )
    jax.block_until_ready(result.density_matrix)
    scf_seconds = time.perf_counter() - scf_started
    if not result.converged:
        raise RuntimeError("GradSCF RHF did not converge.")
    density, overlap, fock, coeff = map(np.asarray, (
        result.density_matrix, result.overlap_matrix,
        result.fock_matrix, result.mo_coeff))
    checks = {
        "electron_trace_abs": abs(float(np.trace(density @ overlap)) - spec.nelectron),
        "orbital_orthogonality_max_abs": float(np.max(np.abs(
            coeff.T @ overlap @ coeff - np.eye(coeff.shape[1])))),
        "commutator_frobenius": float(np.linalg.norm(
            fock @ density @ overlap - overlap @ density @ fock)),
        "min_overlap_eigenvalue": float(np.linalg.eigvalsh(overlap)[0]),
    }
    if (checks["electron_trace_abs"] > 1e-8
            or checks["orbital_orthogonality_max_abs"] > 1e-8
            or checks["min_overlap_eigenvalue"] <= CONFIG.orthogonalization_eps):
        raise RuntimeError(f"GradSCF density/basis validation failed: {checks}")
    row = {
        "nao": topology.nao, "nelectron": spec.nelectron,
        "symbols": list(spec.symbols),
        "nuclear_charges": np.asarray(spec.charges).tolist(),
        "total_energy_hartree": float(result.total_energy),
        "electronic_energy_hartree": float(result.electronic_energy),
        "nuclear_repulsion_hartree": float(result.nuclear_repulsion),
        "converged": bool(result.converged), "cycles": int(result.cycles),
        "integral_seconds": integral_seconds, "scf_seconds": scf_seconds,
        "integrals_reused": reused,
        **checks,
    }
    if check_pyscf:
        t_ref = time.perf_counter()
        ref, ref_checks = reference_energy(
            spec, basis_name, overlap, np.asarray(h), float(enuc))
        row.update(pyscf_energy_hartree=ref,
                   pyscf_energy_abs_error_hartree=abs(ref - row["total_energy_hartree"]),
                   pyscf_seconds=time.perf_counter() - t_ref,
                   reference_integral_checks=ref_checks)
        if row["pyscf_energy_abs_error_hartree"] > 1e-8:
            raise RuntimeError(f"GradSCF/reference energy mismatch: {row}")
    row["total_seconds"] = time.perf_counter() - started
    return row, shared


def provenance(geometry_path):
    root = Path(gradscf.__file__).resolve().parents[2]
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(root), "status", "--short"],
                           capture_output=True, text=True, check=False)
    files = [
        "src/gradscf/integrals/basis.py", "src/gradscf/integrals/plan.py",
        "src/gradscf/integrals/normalization.py",
        "src/gradscf/integrals/backends/native_compact.py",
        "src/gradscf/scf/rhf.py", "src/gradscf/scf/rks.py",
        "src/gradscf/integrals/_native/lib/libgradscf_integrals.dylib",
    ]
    return {
        "python": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(),
        "gradscf_import": gradscf.__file__, "gradscf_root": str(root),
        "git_commit": git.stdout.strip(), "git_status": dirty.stdout.splitlines(),
        "source_sha256": {f: sha256(root / f) for f in files if (root / f).is_file()},
        "jax": jax.__version__, "jaxlib": __import__("jaxlib").__version__,
        "numpy": np.__version__, "devices": [str(x) for x in jax.devices()],
        "dtype": "float64", "integral_backend": "native", "aosym": "s4",
        "density_fitting": False, "spherical": True,
        "thread_environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "command": [sys.executable, *sys.argv], "script_sha256": sha256(__file__),
        "geometry_sha256": sha256(geometry_path),
        "scf_settings": CONFIG.__dict__,
        "hartree_to_kcal_mol": HARTREE_TO_KCAL_MOL,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", type=Path,
                        default=Path(__file__).with_name("s22_hbond_geometries.json"))
    parser.add_argument("--systems", nargs="+",
                        default=["Water_dimer", "Ammonia_dimer", "Formic_acid_dimer"])
    parser.add_argument("--bases", nargs="+", default=["sto-3g", "3-21g", "def2-svp"])
    parser.add_argument("--output", type=Path, default=Path("bsse_results"))
    parser.add_argument("--check-pyscf", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    geometry = json.loads(args.geometries.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "provenance": provenance(args.geometries),
        "geometry_source": {k: v for k, v in geometry.items() if k != "systems"},
        "component_energies": [], "summary": [],
    }
    components = ("AB", "A", "B", "A_ghost_B", "B_ghost_A")
    for system_name in args.systems:
        system = geometry["systems"][system_name]
        for basis_name in args.bases:
            energies, shared = {}, None
            for component in components:
                reuse = shared if component.endswith(("ghost_B", "ghost_A")) else None
                row, computed = calculate(
                    make_spec(system, component), basis_name, args.check_pyscf, reuse)
                if component == "AB":
                    shared = computed
                row.update(system=system_name, basis=basis_name, component=component,
                           integrals_reused=reuse is not None)
                energies[component] = row["total_energy_hartree"]
                report["component_energies"].append(row)
                print(f"{system_name:18s} {basis_name:9s} {component:10s} "
                      f"E={row['total_energy_hartree']:.12f} Ha "
                      f"cycles={row['cycles']} time={row['total_seconds']:.2f}s", flush=True)
                (args.output / "energies.json").write_text(json.dumps(report, indent=2) + "\n")
            raw = energies["AB"] - energies["A"] - energies["B"]
            cp = energies["AB"] - energies["A_ghost_B"] - energies["B_ghost_A"]
            bsse_a = energies["A"] - energies["A_ghost_B"]
            bsse_b = energies["B"] - energies["B_ghost_A"]
            if min(bsse_a, bsse_b) < -1e-8:
                raise RuntimeError("Negative basis-borrowing energy; inspect SCF branches.")
            summary = {"system": system_name, "basis": basis_name,
                       "dimer_nao": computed["overlap"].shape[0],
                       "raw_kcal_mol": raw * HARTREE_TO_KCAL_MOL,
                       "cp_kcal_mol": cp * HARTREE_TO_KCAL_MOL,
                       "bsse_kcal_mol": (bsse_a + bsse_b) * HARTREE_TO_KCAL_MOL,
                       "bsse_a_kcal_mol": bsse_a * HARTREE_TO_KCAL_MOL,
                       "bsse_b_kcal_mol": bsse_b * HARTREE_TO_KCAL_MOL}
            report["summary"].append(summary)
            print("SUMMARY " + json.dumps(summary), flush=True)
            (args.output / "energies.json").write_text(json.dumps(report, indent=2) + "\n")
            shared = None
            jax.clear_caches()
    report["provenance"]["total_elapsed_seconds"] = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["provenance"]["peak_process_rss_mib"] = (
        rss / 1024**2 if sys.platform == "darwin" else rss / 1024)
    report["provenance"]["peak_rss_scope"] = "whole process, including PySCF if requested"
    if args.check_pyscf:
        import pyscf
        report["provenance"]["pyscf_version"] = pyscf.__version__
    (args.output / "energies.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["summary"][0]))
        writer.writeheader()
        writer.writerows(report["summary"])
    print(f"Complete: {len(report['component_energies'])} component calculations; "
          f"{time.perf_counter() - started:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
