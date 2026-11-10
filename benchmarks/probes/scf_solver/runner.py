"""One independent HF/KS workflow per process, using public production APIs.

Run only inside Slurm: python runner.py calculation.json new_output_directory.
SCF timing includes integral/grid preparation. Imports, final certification and
state-file I/O are outside that interval and remain inside the process guard.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {key:plain(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Scientific runs require Slurm")
    cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= cfg["threads"] <= min(64, len(cpus)):
        parser.error("Requested CPUs exceed the allocation")
    os.sched_setaffinity(0, cpus[:cfg["threads"]])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
        JAX_ENABLE_COMPILATION_CACHE="false", GRADSCF_NATIVE_THREADS=str(cfg["threads"]))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "input.json").write_text(json.dumps(cfg, indent=2)+"\n")
    runtime = dict(python=sys.executable, host=platform.node(), job=os.environ["SLURM_JOB_ID"],
        affinity=sorted(os.sched_getaffinity(0)), device="cpu", dtype="float64",
        probe_sha256=digest(__file__), input_sha256=digest(args.config),
        source_snapshot=os.getenv("GRADSCF_SOURCE_SNAPSHOT"),
        memory_units="bytes, max_memory in decimal MB", time_units="seconds",
        nonfinite_serialization="Nonfinite scalars are null; scientific_pass remains false")

    def save(name, value):
        (args.output/name).write_text(json.dumps(plain(value), indent=2, allow_nan=False)+"\n")

    def terminal(status, notes):
        row = dict(case_id=cfg["case_id"], engine=cfg["engine"], method=cfg["method"],
            basis=cfg["basis"], xc=cfg["xc"], natom=cfg["natom"], status=status,
            scientific_pass=False, notes=notes)
        save("summary.json", row)
        save("runtime.json", runtime)
        return 0

    if cfg["method"] in ("RHF", "RKS") and cfg["spin"]:
        return terminal("not_applicable", "Declared open shell; restricted closed-shell method is inapplicable")
    if cfg["engine"] == "gradscf" and cfg["method"] == "GKS" and cfg["xc_family"] == "RSH":
        return terminal("unsupported", "GradSCF spinor RSH is unavailable; no UKS substitution or speed comparison")
    timer = None
    try:
        import numpy as np
        if cfg["engine"] == "gradscf":
            import jax
            import gradscf
            from gradscf import dft, gto, scf
            from gradscf.integrals._native import register_integrals
            from gradscf.integrals.basis import basis_from_molecule_spec
            if cfg.get("solver_api", "shared") == "legacy":
                from gradscf.scf.rks import _mo_residual_norm
                def orbital_gradient_norm(fock, coeff, occupations):
                    if coeff.ndim == 3:
                        return jax.numpy.linalg.norm(jax.vmap(_mo_residual_norm)(fock,coeff,occupations))/2
                    return _mo_residual_norm(fock,coeff,occupations)
            else:
                from gradscf.solvers.scf import orbital_gradient_norm
            jax.config.update("jax_enable_x64", True)
            native = Path(register_integrals()._name).resolve()
            package = Path(gradscf.__file__).parent
            runtime.update(module=gradscf.__file__, jax=jax.__version__, numpy=np.__version__,
                native=str(native), native_sha256=digest(native),
                gradient_certificate=cfg.get("solver_api", "shared"),
                source_sha256={str(p.relative_to(package)):digest(p) for p in
                    list((package/"solvers/scf").glob("*.py"))+list((package/"scf").glob("*.py"))})
            mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], charge=cfg["charge"],
                        spin=cfg["spin"], unit=cfg["unit"], cart=True)
            nao = basis_from_molecule_spec(mol.to_spec(), basis=cfg["basis"]).nao
        else:
            import pyscf
            from pyscf import dft, gto, lib, scf
            lib.num_threads(cfg["threads"])
            runtime.update(module=pyscf.__file__, pyscf=pyscf.__version__, numpy=np.__version__,
                           openmp_threads=lib.num_threads(), libxc=dft.libxc.__version__)
            mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], charge=cfg["charge"],
                        spin=cfg["spin"], unit=cfg["unit"], cart=True, verbose=0, symmetry=False)
            nao = mol.nao_nr()
        runtime["nao"] = int(nao)
        representation = cfg["representation"]
        if representation == "auto":
            representation = ("packed" if cfg["method"].startswith("G") else
                              "direct" if cfg["xc_family"] == "RSH" and cfg["method"].endswith("KS") else
                              "df" if nao > cfg["ri_ao_threshold"] else "packed")
        runtime["actual_representation"] = representation
        pair = nao*(nao+1)//2
        runtime["representation_forecast_bytes"] = dict(full=8*nao**4, s4=8*pair**2, s8=4*pair*(pair+1))
        minimum_bytes = 8*pair**2 if cfg["engine"] == "gradscf" else 4*pair*(pair+1)
        if representation == "packed" and minimum_bytes > cfg["max_memory"]*1e6:
            return terminal("resource_limited", "Known AO representation alone exceeds the registered planning budget")
        save("runtime.json", runtime)
        records = []
        for repeat in range(cfg["repeats"]):
            def timed_out():
                save("timeout.json", dict(repeat=repeat, scope="one SCF facade call",
                    limit_seconds=cfg["single_call_limit_seconds"]))
                os._exit(124)
            timer = threading.Timer(cfg["single_call_limit_seconds"], timed_out)
            timer.daemon = True
            timer.start()
            started = time.perf_counter()
            if cfg["engine"] == "gradscf":
                mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], charge=cfg["charge"],
                            spin=cfg["spin"], unit=cfg["unit"], cart=True)
                factory = getattr(scf if cfg["method"].endswith("HF") else dft, cfg["method"])
                options = dict(max_cycle=cfg["max_cycle"], conv_tol=cfg["conv_tol"],
                    conv_tol_density=cfg["density_tol"], conv_tol_grad=cfg["gradient_tol"],
                    max_memory=cfg["max_memory"], grids_level=3, execution_device="cpu", init_guess=cfg.get("init_guess","hcore"),
                    grid_block_size=-1 if cfg["method"].startswith("G") else 4096)
                if cfg["method"].endswith("KS"):
                    options["xc"] = cfg["xc"]
                mf = factory(mol, **options)
                if representation == "df":
                    mf.density_fit(cfg["auxbasis"])
                elif representation == "direct":
                    mf.direct_scf_workers, mf.direct_scf_tol = cfg["threads"], 0.
                    mf.direct_scf()
                mf.run()
                jax.block_until_ready((mf.e_tot, mf.mo_coeff, mf.mo_energy))
            else:
                mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], charge=cfg["charge"], spin=cfg["spin"],
                            unit=cfg["unit"], cart=True, verbose=0, symmetry=False)
                factory = getattr(scf if cfg["method"].endswith("HF") else dft, cfg["method"])
                mf = factory(mol)
                if cfg["method"].endswith("KS"):
                    mf.xc, mf.grids.level, mf.small_rho_cutoff = cfg["pyscf_xc"], 3, 0.
                    if cfg["method"] == "GKS":
                        mf.collinear = "col"
                mf.max_memory, mf.max_cycle = cfg["max_memory"], cfg["max_cycle"]
                mf.conv_tol, mf.conv_tol_grad = cfg["conv_tol"], cfg["gradient_tol"]
                mf.init_guess, mf.direct_scf_tol = cfg.get("init_guess","hcore"), 0.
                if representation == "df":
                    mf = mf.density_fit(auxbasis=cfg["auxbasis"])
                gates = []

                def check(env):
                    delta = float(abs(env["e_tot"]-env["last_hf_e"]))
                    rms = float(np.sqrt(np.mean(np.abs(env["dm"]-env["dm_last"])**2)))
                    gradient = float(env["norm_gorb"])
                    gates.append(dict(energy_change=delta, density_rms=rms, gradient=gradient))
                    return (np.isfinite(delta+rms+gradient) and delta < cfg["conv_tol"]
                            and rms < cfg["density_tol"] and gradient < cfg["gradient_tol"])

                mf.check_convergence = check  # Same three registered gates, including the final extra cycle.
                mf.run()
            seconds = time.perf_counter()-started
            timer.cancel()
            coeff, occ = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
            overlap = np.asarray(mf.get_ovlp()) if cfg["engine"] == "pyscf" else None
            if cfg["engine"] == "gradscf":
                state = mf.scf_result if mf.scf_result is not None else mf.reference
                overlap = np.asarray(state.overlap_matrix)
                if cfg.get("solver_api", "shared") == "legacy" and cfg["method"].startswith("G"):
                    gradient = float(state.orbital_gradient_norm)
                else:
                    gradient = float(orbital_gradient_norm(state.fock_matrix, mf.mo_coeff, mf.mo_occ))
                inputs = getattr(mf, "_scf_inputs", None)
                weights = getattr(inputs, "grid_weights", None)
                if weights is None:
                    weights = getattr(getattr(state, "grid", None), "weights", None)
                ngrid = (0 if cfg["method"].endswith("HF") else None) if weights is None else int(weights.size)
                factors = getattr(state, "df_factors", None)
                if factors is None:
                    factors = getattr(inputs, "df_factors", None)
                naux = None if factors is None else int(factors.shape[0])
            else:
                gradient = gates[-1]["gradient"] if gates else None
                weights = getattr(getattr(mf, "grids", None), "weights", None)
                ngrid = 0 if weights is None else int(weights.size)
                naux = int(mf.with_df.get_naoaux()) if representation == "df" else None
            stored_eri = getattr(mf, "_eri", None) if cfg["engine"] == "pyscf" else None
            if representation == "df":
                strategy = "auxiliary_ri"
            elif cfg["engine"] == "pyscf":
                strategy = ("packed_plus_range_direct" if stored_eri is not None and cfg["xc_family"] == "RSH" else
                            "packed_incore" if stored_eri is not None else "direct_or_transient_no_retained_eri")
            else:
                strategy = "shell_direct" if representation == "direct" else "packed_incore"
            finite = bool(np.isfinite(mf.e_tot) and np.isfinite(coeff).all()
                          and gradient is not None and np.isfinite(gradient))
            passed = bool(mf.converged and finite and gradient < cfg["gradient_tol"])
            error = coeff.conj().swapaxes(-1,-2) @ overlap @ coeff-np.eye(coeff.shape[-1])
            row = dict(case_id=cfg["case_id"], engine=cfg["engine"], method=cfg["method"], basis=cfg["basis"],
                xc=cfg["xc"], xc_family=cfg["xc_family"], natom=cfg["natom"], nao=int(nao),
                init_guess=cfg.get("init_guess","hcore"),
                nocc=np.count_nonzero(occ, axis=-1).tolist(), nmo=int(coeff.shape[-1]), ngrid=ngrid, naux=naux,
                representation=representation,
                execution_strategy_actual=strategy,
                retained_eri_bytes=None if stored_eri is None else int(stored_eri.nbytes),
                approximation="explicit auxiliary RI" if representation == "df" else "conventional Coulomb, no auxiliary RI",
                auxbasis=cfg["auxbasis"] if representation == "df" else None,
                repeat=repeat, cache_state="fresh process; basis-dimension preflight performed" if repeat == 0 else "new objects, fixed geometry, program reuse only",
                scf_facade_seconds=seconds, energy_hartree=float(mf.e_tot),
                gradient_norm=gradient, orthogonality_error=float(np.linalg.norm(error)),
                converged=bool(mf.converged), cycles=int(mf.cycles), scientific_pass=passed,
                status="ok" if passed else "nonconverged", device="cpu", dtype="float64", threads=cfg["threads"],
                ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                finite_outputs=finite,
                notes="SCF timing includes integral/grid preparation after basis preflight; imports/diagnostic/state I/O excluded. PySCF may retain conventional ERIs despite direct_scf=True; actual strategy is reported. Use guarded process wall time for standalone cost.")
            if cfg["engine"] == "pyscf":
                row["final_gate"] = gates[-1] if gates else None
            np.savez_compressed(args.output/f"state-{repeat}.npz", mo_coeff=coeff, mo_occ=occ,
                                mo_energy=np.asarray(mf.mo_energy), overlap=overlap)
            records.append(row)
            with (args.output/"measurements.jsonl").open("a") as stream:
                stream.write(json.dumps(plain(row), allow_nan=False)+"\n")
            save("summary.json", row)
            if cfg["engine"] == "gradscf":
                del state, inputs, factors
            del mf, mol, coeff, occ, overlap
            del error, weights
            gc.collect()
            if not passed:
                break
        return 0
    except Exception as error:
        if timer is not None:
            timer.cancel()
        traceback.print_exc()
        note = str(error)
        status = ("unsupported" if isinstance(error, (NotImplementedError, ImportError)) else
                  "resource_limited" if isinstance(error, MemoryError) or "memory" in note.lower() else "error")
        return terminal(status, f"{type(error).__name__}: {note}")


if __name__ == "__main__":
    raise SystemExit(main())
