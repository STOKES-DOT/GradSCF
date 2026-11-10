"""CPU integral preparation -> explicit GPU transfer -> production GKS.

Use --config calculation.json --output fresh-dir --budget-seconds 850
--reference cpu-summary.json. Optional --fixed-cycles is never a scientific
pass. --kernels-only --state cpu-state.npz instead diagnoses J/K/XC/eigh at
saved mo_coeff/mo_occ (optional fock_matrix), with independent JIT boundaries.
No CPU fallback, precision reduction, new SCF loop, or derivative is introduced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

from profile_stages import digest, leaves, plain, peak_rss_bytes, synchronize


def array_report(value, device=None):
    rows = [dict(shape=list(a.shape), dtype=str(a.dtype), bytes=int(a.nbytes),
                 devices=sorted(str(d) for d in a.devices()))
            for a in leaves(value) if hasattr(a, "devices")]
    if device is not None and any(r["devices"] != [str(device)] for r in rows):
        raise RuntimeError(f"Array residency mismatch: expected {device}; {rows}")
    if any(r["dtype"] not in {"float64", "complex128"} for r in rows):
        raise RuntimeError(f"Unexpected numerical precision: {rows}")
    return rows


def gpu_capacity(device):
    """Map selected CUDA ordinal through CUDA_VISIBLE_DEVICES; never guess free RAM."""
    query = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free", "--format=csv,noheader,nounits"]
    raw = subprocess.check_output(query, text=True)
    rows = [dict(index=v[0], uuid=v[1], pci_bus_id=v[2], total_bytes=int(v[3])*1024**2,
                 free_bytes=int(v[4])*1024**2)
            for line in raw.splitlines() if (v := [s.strip() for s in line.split(",")])]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ordinal = int(device.local_hardware_id)
    key = visible.split(",")[ordinal].strip() if visible else str(ordinal)
    # CUDA_DEVICE_ORDER=PCI_BUS_ID is set before JAX imports, so numeric CUDA
    # selectors use PCI order, independently of nvidia-smi's index numbering.
    matches = [sorted(rows, key=lambda r: r["pci_bus_id"])[int(key)]] if key.isdigit() else [r for r in rows if r["uuid"].startswith(key)]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot map GPU ordinal {ordinal}, CUDA_VISIBLE_DEVICES={visible!r}")
    return dict(matches[0], cuda_visible_devices=visible, ordinal=ordinal)


def run_kernels(arrays, density, fock, config, jax, event, deadline):
    """Public packed JK and production XC/eigh; no host conversion inside tracing."""
    import jax.numpy as jnp
    import numpy as np
    from gradscf.integrals.molecular.jk import build_jk_from_packed
    from gradscf.scf.gks import _generalized_xc, _hermitian, _validate_mode, _spin_diagonal
    from gradscf.solvers.scf import orbital_orthogonalizer
    n = arrays["overlap"].shape[0]
    kind = _validate_mode(config)
    x = _spin_diagonal(orbital_orthogonalizer(arrays["overlap"], config.orthogonalization_eps))
    def diagonalize(f, orth):
        e, c = jnp.linalg.eigh(_hermitian(orth.conj().T @ f @ orth))
        return e, orth @ c
    cases = [
        ("j_only", lambda e, d: build_jk_from_packed(e, d, with_k=False),
         (arrays["eri"], (density[:n, :n]+density[n:, n:]).T)),
        ("k_only_four_spin_blocks", lambda e, d: build_jk_from_packed(e, d, with_j=False),
         (arrays["eri"], density.reshape(2, n, 2, n).transpose(0, 2, 1, 3))),
        ("xc", lambda d, a, da, w: _generalized_xc(d, a, da, w, config, kind),
         (density, arrays["ao"], arrays["ao_deriv1"], arrays["grid_weights"])),
        ("eigh", diagonalize, (fock, x))]
    rows = []
    for name, fn, arguments in cases:
        row = dict(name=name, status="started", warm_target=3)
        event("kernel_begin", **row)
        if deadline-time.perf_counter() < 5:
            row["status"] = "skipped_budget"
        else:
            synchronize(arguments, jax)
            tick = time.perf_counter()
            lowered = jax.jit(fn).lower(*arguments)
            row["lower_seconds"] = time.perf_counter()-tick
            event("kernel_phase", name=name, phase="lower", seconds=row["lower_seconds"])
            tick = time.perf_counter()
            executable = lowered.compile()
            row["compile_seconds"] = time.perf_counter()-tick
            event("kernel_phase", name=name, phase="compile", seconds=row["compile_seconds"])
            memory = executable.memory_analysis()
            row["compiler_memory_not_measured"] = {k: int(getattr(memory, k)) for k in
                ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")
                if getattr(memory, k, None) is not None}
            tick = time.perf_counter()
            value = executable(*arguments)
            synchronize(value, jax)
            row["first_execute_seconds"] = time.perf_counter()-tick
            event("kernel_phase", name=name, phase="first_execute", seconds=row["first_execute_seconds"])
            samples = []
            for _ in range(3):
                if deadline-time.perf_counter() < max(1., row["first_execute_seconds"]*1.2):
                    break
                tick = time.perf_counter()
                value = executable(*arguments)
                synchronize(value, jax)
                samples.append(time.perf_counter()-tick)
            row.update(warm_seconds=samples, warm_median_seconds=float(np.median(samples)) if samples else None,
                       status="ok" if len(samples) == 3 else "partial_budget",
                       output_arrays=array_report(value),
                       finite=all(bool(np.isfinite(np.asarray(a)).all()) for a in leaves(value) if hasattr(a, "shape")))
            row["snapshot_norms"] = [dict(norm=float(np.linalg.norm(np.asarray(a))),
                max_abs=float(np.max(np.abs(np.asarray(a))))) for a in leaves(value) if hasattr(a, "shape")]
        rows.append(row)
        event("kernel_end", **row)
    return rows


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-seconds", type=float, default=850.)
    parser.add_argument("--fixed-cycles", type=int)
    parser.add_argument("--reference", type=Path, help="CPU summary.json with energy_hartree")
    parser.add_argument("--state", type=Path, help="CPU mo_coeff/mo_occ npz, required with --kernels-only")
    parser.add_argument("--kernels-only", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0, help="Visible CUDA ordinal")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if not 0 < args.budget_seconds <= 1800 or (args.fixed_cycles is not None and args.fixed_cycles < 1) or args.gpu_index < 0:
        parser.error("Budget must be (0,1800], fixed cycles positive")
    if cfg.get("method") != "GKS" or cfg.get("representation") not in {"auto", "packed"}:
        parser.error("Only conventional packed GKS is supported")
    if cfg.get("init_guess", "hcore") != "hcore" or args.kernels_only and args.state is None:
        parser.error("Full SCF requires hcore; --kernels-only requires --state")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Scientific runs require Slurm")
    cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= cfg.get("threads", 8) <= min(64, len(cpus)):
        parser.error("Requested CPUs exceed the allocation")
    os.sched_setaffinity(0, cpus[:cfg.get("threads", 8)])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(JAX_PLATFORMS="cuda,cpu", JAX_ENABLE_X64="1", JAX_ENABLE_COMPILATION_CACHE="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", CUDA_DEVICE_ORDER="PCI_BUS_ID", GRADSCF_NATIVE_THREADS=str(cfg.get("threads", 8)))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".9")
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output / name).write_text(json.dumps(plain(value), indent=2, allow_nan=False)+"\n")
    lock, stop = threading.Lock(), threading.Event()
    events = (args.output / "events.jsonl").open("a", buffering=1)
    gpu, sampler = None, None
    def memory():
        try:
            rss = int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            rss = None
        try:
            stats = gpu.memory_stats() if gpu is not None else None
        except Exception as exc:
            stats = dict(unavailable=repr(exc))
        return dict(rss_bytes=rss, rss_peak_bytes=peak_rss_bytes(), gpu_memory_stats=stats,
                    gpu_memory_note="JAX bytes_in_use/reserved/peak are allocator counters; not process RSS or global free VRAM")
    def event(kind, **value):
        with lock:
            events.write(json.dumps(plain(dict(event=kind, elapsed_seconds=time.perf_counter()-started,
                                               **value)), allow_nan=False)+"\n")
            events.flush()
    def timeout():
        save("summary.json", dict(status="timeout", scientific_pass=False, budget_seconds=args.budget_seconds))
        event("timeout")
        os._exit(124)
    deadline = started+args.budget_seconds
    timer = threading.Timer(max(.001, deadline-time.perf_counter()), timeout)
    timer.daemon = True
    timer.start()
    runtime = dict(python=sys.executable, python_version=platform.python_version(), pythonpath=sys.path,
        host=platform.node(), config=cfg, input_sha256=digest(args.config), probe_sha256=digest(__file__),
        budget_seconds=args.budget_seconds, precision="float64/complex128", cpu_fallback_allowed=False,
        slurm_job=os.environ["SLURM_JOB_ID"], affinity=sorted(os.sched_getaffinity(0)),
        helper_sha256=digest(Path(__file__).with_name("profile_stages.py")),
        reference=str(args.reference.resolve()) if args.reference else None,
        reference_sha256=digest(args.reference) if args.reference else None,
        state_sha256=digest(args.state) if args.state else None,
        fock_source="production_result" if not args.kernels_only else "pending_state_inspection",
        timing_scope="CPU preparation + synchronized transfer + production forward; GPU SCF includes compile",
        environment={k: v for k, v in os.environ.items() if k.startswith(("JAX_", "XLA_", "GRADSCF_", "CUDA_"))})
    save("runtime.json", runtime)
    try:
        import numpy as np
        import jax
        import jax.numpy as jnp
        import gradscf
        from gradscf import gto
        from gradscf.integrals._native import register_integrals
        from gradscf.scf.gks import GKSConfig, run_gks_from_integrals, _spin_diagonal
        from gradscf.scf.inputs.assembly import build_uks_integral_inputs
        jax.config.update("jax_enable_x64", True)
        cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[args.gpu_index]
        if gpu.platform != "gpu":
            raise RuntimeError("Selected execution device is not GPU")
        native = Path(register_integrals()._name).resolve()
        package = Path(gradscf.__file__).resolve().parent
        try:
            imported_head = subprocess.check_output(["git", "-C", str(package), "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            imported_head = None
        runtime.update(jax=jax.__version__, module=gradscf.__file__, gpu=str(gpu), gpu_kind=gpu.device_kind,
            native=str(native), native_sha256=digest(native), imported_head=imported_head,
            source_sha256={str(p.relative_to(package)): digest(p) for folder in
                ("scf", "solvers/scf", "integrals") for p in (package / folder).rglob("*.py")})
        save("runtime.json", runtime)
        def sample():
            while not stop.wait(.5):
                event("memory_sample", **memory())
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        config = GKSConfig(xc_spec=cfg["xc"], max_cycle=args.fixed_cycles or cfg["max_cycle"],
            conv_tol=0. if args.fixed_cycles else cfg["conv_tol"], conv_tol_density=cfg["density_tol"],
            conv_tol_grad=cfg["gradient_tol"], collinear=cfg.get("collinear", "col"),
            damping=cfg.get("damp", 0.), level_shift=cfg.get("level_shift", 0.),
            max_memory=cfg["max_memory"], grid_block_size=-1, grid_max_memory=cfg.get("grid_max_memory"))
        tick = time.perf_counter()
        event("cpu_preparation_begin", **memory())
        with jax.default_device(cpu):
            mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], charge=cfg["charge"], spin=cfg["spin"],
                        unit=cfg["unit"], cart=True)
            inputs = build_uks_integral_inputs(atom=mol.to_spec(), basis=mol.basis, config=config,
                xc_spec=cfg["xc"], unit=mol.unit, charge=mol.charge, spin=mol.spin, cart=mol.cart,
                grids_level=cfg.get("grid_level", cfg.get("grids_level", 3)), max_l=cfg.get("max_l", 3), init_guess="hcore",
                integral_backend="libcint", grid_ao_backend="jax", libcint_geometry_grad_policy="analytic")
            arrays = {k: getattr(inputs, k) for k in
                      ("overlap", "hcore", "eri", "nuclear_repulsion", "ao", "ao_deriv1", "grid_weights")}
            arrays = {k: jnp.asarray(v) for k, v in arrays.items()}
            state = None
            if args.kernels_only:
                with np.load(args.state, allow_pickle=False) as stored:
                    runtime["fock_source"] = "cpu_state.fock_matrix" if "fock_matrix" in stored else "spin_diagonal_hcore"
                    c, o = jnp.asarray(stored["mo_coeff"], dtype=jnp.complex128), jnp.asarray(stored["mo_occ"], dtype=jnp.float64)
                    state = dict(density=(c*o) @ c.conj().T,
                        fock=jnp.asarray(stored["fock_matrix"]) if "fock_matrix" in stored else _spin_diagonal(arrays["hcore"]))
            synchronize((inputs, arrays, state), jax)
        cpu_seconds = time.perf_counter()-tick
        save("runtime.json", runtime)
        metadata = {k: array_report(v, cpu) for k, v in arrays.items()}
        state_metadata = array_report(state, cpu)
        event("cpu_preparation_end", seconds=cpu_seconds, arrays=metadata, **memory())
        transferred_bytes = sum(a["bytes"] for v in metadata.values() for a in v)+sum(a["bytes"] for a in state_metadata)
        # Retained inputs + at least 2.7 GB compiler/workspace + 3 GB headroom.
        estimate = transferred_bytes+max(2_700_000_000, int(transferred_bytes*.15))+3_000_000_000
        capacity = gpu_capacity(gpu)
        stats = gpu.memory_stats() or {}
        pool_available = stats.get("bytes_limit", capacity["free_bytes"])-stats.get("bytes_in_use", 0)
        capacity.update(jax_pool_available_bytes=pool_available,
                        usable_bytes=min(capacity["free_bytes"], pool_available))
        event("gpu_memory_preflight", capacity=capacity, required_bytes_estimate=estimate,
              input_bytes=transferred_bytes, estimate_note="Conservative preflight, not an OOM guarantee")
        if capacity["usable_bytes"] < estimate:
            save("summary.json", dict(status="resource_limited", scientific_pass=False, capacity=capacity,
                                      required_bytes_estimate=estimate, cpu_preparation_seconds=cpu_seconds))
            return 0
        tick = time.perf_counter()
        with jax.default_device(gpu):
            arrays, state = jax.tree_util.tree_map(lambda a: jax.device_put(a, gpu), (arrays, state))
            synchronize((arrays, state), jax)
        transfer_seconds = time.perf_counter()-tick
        event("gpu_transfer_end", seconds=transfer_seconds, bytes=transferred_bytes,
              arrays=array_report((arrays, state), gpu), **memory())
        tick = time.perf_counter()
        with jax.default_device(gpu):
            if args.kernels_only:
                summary = dict(status="kernels_only", scientific_pass=False)
            else:
                event("gpu_scf_begin", **memory())
                result = run_gks_from_integrals(**arrays, nelectron=mol.nelectron, config=config)
                synchronize(result, jax)
                elapsed = time.perf_counter()-tick
                array_report(result, gpu)
                reference = json.loads(args.reference.read_text()) if args.reference else {}
                reference_valid = bool(reference.get("converged") and not reference.get("fixed_cycle_probe", False))
                delta = float(result.total_energy)-reference["energy_hartree"] if reference else None
                tolerance = cfg.get("energy_comparison_tolerance_hartree", 1e-7)
                finite = bool(np.isfinite(result.total_energy) and np.isfinite(result.orbital_gradient_norm))
                passed = bool(args.fixed_cycles is None and result.converged and finite and reference_valid and
                    result.orbital_gradient_norm < cfg["gradient_tol"] and delta is not None and abs(delta) <= tolerance)
                summary = dict(status="fixed_cycle_probe" if args.fixed_cycles else "ok" if result.converged else "nonconverged",
                    scientific_pass=passed, converged=bool(result.converged), cycles=int(result.cycles),
                    energy_hartree=float(result.total_energy), gradient_norm=float(result.orbital_gradient_norm),
                    gpu_scf_seconds=elapsed, energy_delta_cpu_hartree=delta, energy_tolerance_hartree=tolerance,
                    reference_status="compared" if reference else "missing", gradient_tolerance=cfg["gradient_tol"])
                summary["reference_qualified_converged"] = reference_valid
                event("gpu_scf_end", **summary, **memory())
                state = dict(density=result.density_matrix, fock=result.fock_matrix)
            summary.update(cpu_preparation_seconds=cpu_seconds, transfer_seconds=transfer_seconds,
                           transferred_bytes=transferred_bytes, nao=int(inputs.overlap.shape[0]))
            save("summary.json", summary)
            summary["kernels"] = run_kernels(arrays, state["density"], state["fock"], config, jax, event, deadline)
        summary.update(process_seconds=time.perf_counter()-started, memory=memory())
        save("summary.json", summary)
        return 0
    except Exception as exc:
        traceback.print_exc()
        save("failure.json", dict(status="error", scientific_pass=False, error=repr(exc), **memory()))
        return 1
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=1.)
        timer.cancel()
        events.close()


if __name__ == "__main__":
    raise SystemExit(main())
