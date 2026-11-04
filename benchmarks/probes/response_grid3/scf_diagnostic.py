"""SCF-only convergence evidence, without changing the production iteration.

Run inside Slurm: python scf_diagnostic.py case.json output [--engine pyscf].
Callbacks add scalar telemetry, so these timings are diagnostic, not benchmarks.
The original case thresholds, geometry, basis, grid and hcore guess are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--engine", choices=("gradscf", "pyscf"), default="gradscf")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Numerical runs require Slurm")
    cfg = json.loads(args.config.read_text())
    if cfg["ks"] != "uks" or cfg["representation"] != "df":
        parser.error("This narrow diagnosis expects the archived UKS/RI case")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < cfg["threads"]:
        parser.error("Insufficient allocated CPUs")
    os.sched_setaffinity(0, affinity[:cfg["threads"]])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cfg["blas_threads"])
    os.environ["GRADSCF_NATIVE_THREADS"] = str(cfg["threads"])
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "input.json").write_text(json.dumps(cfg, indent=2) + "\n")
    events = (args.output / "iterations.jsonl").open("w", buffering=1)
    start = time.monotonic()

    def record(kind, **values):
        row = dict(kind=kind, elapsed_seconds=time.monotonic()-start, **values)
        events.write(json.dumps(row, allow_nan=False) + "\n")

    import numpy as np
    runtime = dict(engine=args.engine, python=sys.executable,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        job=os.environ["SLURM_JOB_ID"], affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GRADSCF_NATIVE_THREADS")},
        note="Scalar callback instrumentation; no production numerical changes; no clean speed claim")
    if args.engine == "gradscf":
        import jax
        import jax.numpy as jnp
        import gradscf
        from gradscf import dft, gto
        from gradscf.integrals._native import register_integrals
        from gradscf.scf import uks
        from gradscf.solvers.scf import iterate
        jax.config.update("jax_enable_x64", True)
        native = Path(register_integrals()._name).resolve()
        runtime.update(version=gradscf.__file__, jax=jax.__version__,
            native=str(native), native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            scf_source={name: hashlib.sha256(Path(uks.__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("uks.py", "rks.py", "iteration.py")},
            solver_source={name: hashlib.sha256(Path(iterate.__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ("iterate.py", "metrics.py", "types.py")})
        original_scan, original_gate = uks.run_unrestricted_scf, iterate.convergence_reached
        counts = dict(state=0, metric=0)

        def state_event(energy, gaps, populations, spin_squared):
            record("state", cycle=counts["state"], energy_hartree=float(energy),
                gap_hartree=np.asarray(gaps).tolist(), electrons=np.asarray(populations).tolist(),
                spin_squared=float(spin_squared))
            counts["state"] += 1

        def metric_event(de, dd, grad):
            counts["metric"] += 1
            record("metrics", cycle=counts["metric"], energy_change_hartree=float(de),
                density_rms=float(dd), orbital_gradient_norm=float(grad),
                energy_pass=bool(de < cfg["scf_tol"]), density_pass=bool(dd < cfg["density_tol"]),
                gradient_pass=bool(grad < cfg["gradient_tol"]))

        def gate(de, dd, grad, **kwargs):
            jax.debug.callback(metric_event, de, dd, grad, ordered=True)
            return original_gate(de, dd, grad, **kwargs)

        def scan(**kwargs):
            builder, overlap = kwargs.pop("fock_builder"), kwargs["overlap"]
            na = (mol.nelectron + cfg["spin"]) // 2
            nb = mol.nelectron - na

            def observed(density, coeff, energies):
                result = builder(density, coeff, energies)
                gaps = jnp.stack((energies[0, na]-energies[0, na-1],
                                  energies[1, nb]-energies[1, nb-1]))
                populations = jnp.einsum("sij,ji->s", density, overlap)
                cross = coeff[0, :, :na].T @ overlap @ coeff[1, :, :nb]
                s2 = .25*cfg["spin"]**2 + .5*(na+nb) - jnp.sum(cross**2)
                jax.debug.callback(state_event, result[2], gaps, populations, s2, ordered=True)
                return result

            result = original_scan(fock_builder=observed, **kwargs)
            jax.debug.callback(lambda cycle, best, rms: record("selection", selected_cycle=int(cycle),
                best_cycle=int(best), selected_density_rms=float(rms)),
                result[7], result[8], result[9], ordered=True)
            return result

        iterate.convergence_reached, uks.run_unrestricted_scf = gate, scan
        mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], charge=cfg["charge"],
                    spin=cfg["spin"], unit=cfg["unit"], cart=cfg["cart"])
        mf = dft.UKS(mol, xc=cfg["xc"], grids_level=cfg["grid_level"], execution_device="cpu",
            conv_tol=cfg["scf_tol"], conv_tol_density=cfg["density_tol"],
            conv_tol_grad=cfg["gradient_tol"], max_cycle=cfg["max_cycle"],
            max_memory=cfg["max_memory"], grid_block_size=cfg["grid_block_size"], init_guess="hcore")
        mf.density_fit(cfg["auxbasis"])
    else:
        import pyscf
        from pyscf import dft, gto, lib
        lib.num_threads(cfg["threads"])
        runtime.update(version=pyscf.__version__, module=pyscf.__file__,
            reference_note="Independent grid3 PySCF quadrature and convergence policy; not identical discrete inputs")
        mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], charge=cfg["charge"],
                    spin=cfg["spin"], unit=cfg["unit"], cart=cfg["cart"], verbose=4)
        mf = dft.UKS(mol, xc=cfg["xc"]).density_fit(auxbasis=cfg["auxbasis"])
        mf.grids.level = cfg["grid_level"]
        mf.max_memory, mf.max_cycle = cfg["max_memory"], cfg["max_cycle"]
        mf.conv_tol, mf.conv_tol_grad, mf.init_guess = cfg["scf_tol"], cfg["gradient_tol"], "hcore"

        def callback(env):
            energy, density, energies = env["e_tot"], np.asarray(env["dm"]), env["mo_energy"]
            na, nb = mol.nelec
            dd = np.sqrt(np.mean((density-np.asarray(env["dm_last"]))**2))
            record("metrics", cycle=int(env["cycle"])+1, energy_hartree=float(energy),
                energy_change_hartree=float(abs(energy-env["last_hf_e"])), density_rms=float(dd),
                orbital_gradient_norm=float(env["norm_gorb"]),
                gap_hartree=[float(energies[0][na]-energies[0][na-1]), float(energies[1][nb]-energies[1][nb-1])])

        mf.callback = callback
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    t0 = time.perf_counter()
    mf.run()
    if args.engine == "gradscf":
        jax.effects_barrier()
    energy = float(mf.e_tot)
    np.savez(args.output / "scf_state.npz", mo_coeff=np.asarray(mf.mo_coeff),
        mo_energy=np.asarray(mf.mo_energy), mo_occ=np.asarray(mf.mo_occ), density=np.asarray(mf.make_rdm1()))
    status = dict(converged=bool(mf.converged), energy_hartree=energy,
        cycles=int(mf.cycles), wall_seconds=time.perf_counter()-t0,
        ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        timing_scope="SCF facade with diagnostics, including integral/grid preparation")
    (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    record("finished", **status)
    events.close()


if __name__ == "__main__":
    main()
