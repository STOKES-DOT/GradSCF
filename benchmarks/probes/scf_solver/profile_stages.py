"""One fresh-process SCF measurement, with optional invasive diagnostics.

Run clean and diagnostic in separate Slurm processes, with the same JSON used
by runner.py. Example (one facade call, regardless of config['repeats']):

  python profile_stages.py calculation.json clean-out --mode clean --budget-seconds 800
  python profile_stages.py calculation.json diag-out --mode diagnostic --budget-seconds 950

For a deliberately unconverged runtime trace, add --fixed-cycles 5 to diagnostic.
It copies the JSON configuration and sets max_cycle=5, conv_tol=0; the original
input is preserved separately and scientific_pass is always false for this probe.

Diagnostic enables cProfile, JAX compile logs and synchronized hierarchical
Python spans. Add --jax-trace only when the installed JAX profiler works; its
trace is bounded to this one SCF but can still be large. No callbacks are added
to the numerical solver. Python calls made while tracing a lax.scan are NOT
iteration counts or execution timings. Inclusive nested spans are not additive,
and diagnostic timing is not a clean benchmark. Inputs remain in memory only
in diagnostic mode, for run_kernels; no ERI/state tensor is serialized.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import cProfile
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import pstats
import resource
import subprocess
import sys
import threading
import time
import traceback


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def peak_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else 1024 * rss)


def leaves(value):
    # asdict() would deepcopy the multi-GB ERI. Visit references only.
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from leaves(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from leaves(item)
    else:
        yield value


def synchronize(value, jax):
    for leaf in leaves(value):
        if not isinstance(leaf, jax.core.Tracer) and hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def array_metadata(value):
    return {field.name: dict(shape=list(item.shape), dtype=str(item.dtype),
                            bytes=int(item.size * item.dtype.itemsize))
            for field in fields(value)
            if hasattr(item := getattr(value, field.name), "shape")}


def effective_config(original, fixed_cycles=None):
    cfg = dict(original)
    if fixed_cycles is not None:
        cfg.update(max_cycle=fixed_cycles, conv_tol=0.)
    return cfg


class Stages:
    """Temporary hooks at known preparation/solver boundaries; no runtime callbacks."""

    def __init__(self, jax, event, trace=False):
        self.jax, self.event, self.trace = jax, event, trace
        self.stack, self.rows, self.patches = [], [], []
        self.captured = {}
        self.next_id = 0

    def wrapper(self, original, label, capture=None):
        @wraps(original)
        def measured(*args, **kwargs):
            name = label(args, kwargs) if callable(label) else label
            call_id, parent_id = self.next_id, self.stack[-1] if self.stack else None
            self.next_id += 1
            traced = any(isinstance(x, self.jax.core.Tracer) for x in leaves((args, kwargs)))
            row = dict(call_id=call_id, parent_id=parent_id, stage=name,
                       depth=len(self.stack), tracing=traced)
            self.event("stage_begin", **row)
            self.stack.append(call_id)
            started = time.perf_counter()
            annotation = (self.jax.profiler.TraceAnnotation(name, python_call_id=call_id)
                          if self.trace else nullcontext())
            try:
                if name == "solver.solve_scf":
                    kwargs = dict(kwargs)
                    for callback_name in ("evaluate", "diagonalize", "density_from_coeff",
                                          "diis_error", "orbital_gradient", "shift_fock"):
                        if kwargs.get(callback_name) is not None:
                            kwargs[callback_name] = self.scoped_callback(kwargs[callback_name], callback_name)
                with annotation, self.jax.named_scope(name):
                    result = original(*args, **kwargs)
                    traced = traced or any(isinstance(x, self.jax.core.Tracer) for x in leaves(result))
                    if not traced:
                        synchronize(result, self.jax)
                        if capture:
                            self.captured[capture] = result
                row.update(status="ok", tracing=traced)
                return result
            except BaseException as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                row.update(seconds=time.perf_counter() - started,
                           timing_kind="python_tracing_only" if traced else "synchronized_inclusive",
                           ru_maxrss_bytes=peak_rss_bytes())
                self.stack.pop()
                self.rows.append(row)
                self.event("stage_end", **row)
        return measured

    def scoped_callback(self, original, name):
        @wraps(original)
        def scoped(*args, **kwargs):
            with self.jax.named_scope("scf." + name):
                return original(*args, **kwargs)
        return scoped

    def patch(self, owner, name, label, capture=None, aliases=True):
        original = getattr(owner, name)
        wrapped = self.wrapper(original, label, capture)
        targets = [(owner, name)]
        if aliases:
            # Production modules import these functions by value. Patch every
            # already-loaded alias, then restore all bindings before microbenchmarks.
            targets = [(module, key) for module in list(sys.modules.values())
                       if getattr(module, "__name__", "").startswith("gradscf.")
                       for key, value in list(vars(module).items()) if value is original]
        for target, key in targets:
            self.patches.append((target, key, original))
            setattr(target, key, wrapped)

    def install(self, representation):
        module = lambda name: importlib.import_module("gradscf." + name)
        basis, assembly = module("integrals.basis"), module("scf.inputs.assembly")
        grid, ao, plan = module("scf.inputs.grid"), module("integrals.grids.ao"), module("integrals.plan")
        rks, uks, gks = module("scf.rks"), module("scf.uks"), module("scf.gks")
        jk, solver = module("integrals.molecular.jk"), module("solvers.scf.iterate")
        for owner, names in (
            (basis, ("prepare_basis", "basis_from_molecule_spec")),
            (assembly, ("_build_common", "_prepare_basis_grid_context", "_grid_ao_payload")),
            (grid, ("_cached_grid_ao_input_bundle",)),
            (module("integrals.grids"), ("build_molecular_grid_from_spec",)),
            (ao, ("evaluate_cartesian_ao_with_derivatives",)),
            (gks, ("generalized_energy_and_fock", "generalized_jk", "_generalized_xc")),
            (uks, ("_raw_fock_and_energy_for_state", "_spin_jk_matrices",
                   "_unrestricted_xc_energy_and_potential_on_grid", "_unrestricted_vxc_matrices")),
            (rks, ("_fock_components_for_density", "_xc_energy_and_potential_on_grid",
                   "_vxc_matrix_from_grid_potential")),
            (jk, ("build_jk_from_packed", "build_jk_from_df", "build_jk_from_df_orbitals")),
            (module("solvers.nonlinear.diis"), ("diis_extrapolate",)),
        ):
            for name in names:
                self.patch(owner, name, f"{owner.__name__}.{name}")
        for name in ("build_rks_integral_inputs", "build_uks_integral_inputs"):
            self.patch(assembly, name, f"inputs.{name}", capture="inputs")
        for owner, name in ((rks, "run_rks_from_integrals"), (uks, "run_uks_from_integrals"),
                            (gks, "run_gks_from_integrals")):
            self.patch(owner, name, f"solver.{name}", capture="result")
        self.patch(solver, "solve_scf", "solver.solve_scf", capture="shared_result")
        self.patch(plan.IntegralPlan, "evaluate", lambda args, kw:
                   f"integrals.{args[1] if len(args) > 1 else kw['operator']}.{kw.get('aosym', 's1')}",
                   aliases=False)
        if representation == "df":
            df = module("integrals.molecular.density_fitting")
            self.patch(df.AuxiliaryPlan, "evaluate", "integrals.auxiliary_2c_3c", aliases=False)
            self.patch(df.AuxiliaryPlan, "factors", "integrals.auxiliary_factors", aliases=False)
            self.patch(df, "unpack_factors", "integrals.unpack_ri_factors")
        self.event("hooks_installed", bindings=len(self.patches))

    def restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def summary(self):
        groups = defaultdict(lambda: dict(python_calls=0, inclusive_seconds=0.))
        for row in self.rows:
            group = groups[(row["stage"], row["timing_kind"], row["status"])]
            group["python_calls"] += 1
            group["inclusive_seconds"] += row["seconds"]
        return [dict(stage=key[0], timing_kind=key[1], status=key[2], **value)
                for key, value in sorted(groups.items())]


def main(argv=None):
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", required=True, choices=("clean", "diagnostic"))
    parser.add_argument("--budget-seconds", type=float, default=1800.)
    parser.add_argument("--jax-trace", action="store_true", help="Optional potentially large, one-SCF JAX trace")
    parser.add_argument("--fixed-cycles", type=int,
                        help="Diagnostic only: fixed N-cycle snapshot with early convergence disabled")
    args = parser.parse_args(argv)
    original_cfg = json.loads(args.config.read_text())
    if not 0 < args.budget_seconds <= 1800:
        parser.error("--budget-seconds must be in (0, 1800]")
    if args.jax_trace and args.mode != "diagnostic":
        parser.error("--jax-trace requires --mode diagnostic")
    if args.fixed_cycles is not None and (args.mode != "diagnostic" or args.fixed_cycles < 1):
        parser.error("--fixed-cycles requires diagnostic mode and a positive cycle count")
    cfg = effective_config(original_cfg, args.fixed_cycles)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Scientific runs require Slurm")
    if cfg.get("engine") != "gradscf":
        parser.error("This probe profiles GradSCF only")
    cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= cfg["threads"] <= min(64, len(cpus)):
        parser.error("Requested CPUs exceed the allocation")
    os.sched_setaffinity(0, cpus[:cfg["threads"]])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
        JAX_ENABLE_COMPILATION_CACHE="false", GRADSCF_NATIVE_THREADS=str(cfg["threads"]))
    os.environ["JAX_LOG_COMPILES"] = "1" if args.mode == "diagnostic" else "0"
    args.output.mkdir(parents=True, exist_ok=False)
    args.output = args.output.resolve()
    (args.output / "input.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.fixed_cycles is not None:
        (args.output / "original_input.json").write_text(args.config.read_text())
    # FD redirection includes native stderr/stdout. Python writes are line-buffered.
    sys.stdout.flush()
    sys.stderr.flush()
    raw = (args.output / "raw.log").open("a", buffering=1)
    os.dup2(raw.fileno(), 1)
    os.dup2(raw.fileno(), 2)
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
    events = (args.output / "events.jsonl").open("a", buffering=1)
    event_lock = threading.Lock()

    def save(name, value):
        (args.output / name).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")

    def event(kind, **data):
        row = dict(event=kind, elapsed_seconds=time.perf_counter() - process_start,
                   utc=datetime.now(timezone.utc).isoformat(), **data)
        with event_lock:
            events.write(json.dumps(plain(row), allow_nan=False) + "\n")
            events.flush()
        print(json.dumps(plain(row), allow_nan=False), flush=True)

    deadline = process_start + args.budget_seconds
    runtime = dict(mode=args.mode, python=sys.executable, python_version=platform.python_version(),
        platform=platform.platform(), host=platform.node(), job=os.environ["SLURM_JOB_ID"],
        affinity=sorted(os.sched_getaffinity(0)), budget_seconds=args.budget_seconds,
        probe_sha256=digest(__file__), input_sha256=digest(args.config),
        effective_input_sha256=digest(args.output / "input.json"),
        fixed_cycle_probe=args.fixed_cycles is not None, fixed_cycles=args.fixed_cycles,
        original_max_cycle=original_cfg["max_cycle"], original_conv_tol=original_cfg["conv_tol"],
        source_snapshot=os.getenv("GRADSCF_SOURCE_SNAPSHOT"),
        environment={key: value for key, value in os.environ.items() if key.startswith((
            "JAX_", "XLA_", "GRADSCF_", "OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_"))},
        configured_repeats=cfg.get("repeats"), measured_facade_calls=1,
        time_units="seconds", memory_units="bytes; max_memory is decimal MB",
        timing_scope="facade construction/run including integral/grid preparation; imports/preflight excluded",
        interpretation="Diagnostic spans include synchronization/profiler/logging overhead; nested inclusive spans are not additive. Tracing calls are not iteration counts.",
        retention="Diagnostic captures existing input/result references through kernels; no deep copies or tensor serialization. This extends ERI lifetime beyond facade return.")
    save("runtime.json", runtime)
    event("process_begin", mode=args.mode, config=str(args.config.resolve()))

    def timed_out(scope):
        row = dict(status="timeout", scope=scope, budget_seconds=args.budget_seconds,
                   process_seconds=time.perf_counter() - process_start, ru_maxrss_bytes=peak_rss_bytes())
        save("timeout.json", row)
        event("timeout", **row)
        os._exit(124)

    timer = threading.Timer(max(.001, deadline - time.perf_counter()), timed_out, args=("whole process",))
    timer.daemon = True
    timer.start()
    stages, profile, trace_active, call_timer = None, None, False, None
    try:
        import_start = time.perf_counter()
        event("imports_begin")
        import numpy as np
        import jax
        import gradscf
        from gradscf import dft, gto, scf
        from gradscf.integrals._native import register_integrals
        from gradscf.integrals.basis import basis_from_molecule_spec
        from gradscf.solvers.scf import orbital_gradient_norm
        jax.config.update("jax_enable_x64", True)
        native = Path(register_integrals()._name).resolve()
        package = Path(gradscf.__file__).resolve().parent
        try:
            imported_head = subprocess.check_output(["git", "-C", str(package), "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            imported_head = None
        runtime.update(module=gradscf.__file__, imported_head=imported_head, native=str(native),
            native_sha256=digest(native), jax=jax.__version__, numpy=np.__version__,
            devices=[str(device) for device in jax.devices()], x64=bool(jax.config.jax_enable_x64),
            source_sha256={str(path.relative_to(package)): digest(path) for path in
                list((package / "scf").rglob("*.py")) + list((package / "solvers/scf").glob("*.py")) +
                list((package / "integrals").rglob("*.py"))})
        runtime["imports_seconds"] = time.perf_counter() - import_start
        event("imports_end", seconds=runtime["imports_seconds"], module=gradscf.__file__)
        started = time.perf_counter()
        event("basis_preflight_begin")
        mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], charge=cfg["charge"],
                    spin=cfg["spin"], unit=cfg["unit"], cart=True)
        preflight_basis = basis_from_molecule_spec(mol.to_spec(), basis=cfg["basis"])
        synchronize(preflight_basis, jax)
        nao = preflight_basis.nao
        runtime["basis_preflight_seconds"] = time.perf_counter() - started
        runtime["nao"] = int(nao)
        representation = cfg["representation"]
        if representation == "auto":
            representation = ("packed" if cfg["method"].startswith("G") else
                "direct" if cfg["xc_family"] == "RSH" and cfg["method"].endswith("KS") else
                "df" if nao > cfg["ri_ao_threshold"] else "packed")
        if representation not in {"packed", "df", "direct"}:
            raise ValueError(f"Unsupported representation: {representation}")
        runtime["actual_representation"] = representation
        event("basis_preflight_end", seconds=runtime["basis_preflight_seconds"], nao=int(nao))
        save("runtime.json", runtime)
        if args.mode == "diagnostic":
            stages = Stages(jax, event)
            stages.install(representation)
            profile = cProfile.Profile()
            if args.jax_trace:
                try:
                    jax.profiler.start_trace(str(args.output / "jax-trace"))
                    trace_active = stages.trace = True
                    event("jax_trace_started")
                except Exception as error:
                    event("jax_trace_unavailable", error=f"{type(error).__name__}: {error}")
        options = dict(max_cycle=cfg["max_cycle"], conv_tol=cfg["conv_tol"],
            conv_tol_density=cfg["density_tol"], conv_tol_grad=cfg["gradient_tol"],
            max_memory=cfg["max_memory"], grids_level=3, execution_device="cpu",
            init_guess=cfg.get("init_guess", "hcore"),
            grid_block_size=-1 if cfg["method"].startswith("G") else 4096)
        if cfg["method"].endswith("KS"):
            options["xc"] = cfg["xc"]
        factory = getattr(scf if cfg["method"].endswith("HF") else dft, cfg["method"])
        call_limit = min(float(cfg.get("single_call_limit_seconds", args.budget_seconds)),
                         deadline - time.perf_counter())
        if call_limit <= 0:
            raise TimeoutError("Process budget exhausted before facade call")
        call_timer = threading.Timer(call_limit, timed_out, args=("one SCF facade call",))
        call_timer.daemon = True
        call_timer.start()
        event("facade_begin", options=options, call_limit_seconds=call_limit)
        if profile:
            profile.enable()
        started = time.perf_counter()
        mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], charge=cfg["charge"],
                    spin=cfg["spin"], unit=cfg["unit"], cart=True)
        mf = factory(mol, **options)
        if representation == "df":
            mf.density_fit(cfg["auxbasis"])
        elif representation == "direct":
            mf.direct_scf_workers, mf.direct_scf_tol = cfg["threads"], 0.
            mf.direct_scf()
        mf.run()
        # Synchronize every retained result array without copying it to host.
        result = mf.scf_result if mf.scf_result is not None else mf.reference
        synchronize((mf.e_tot, mf.mo_coeff, mf.mo_energy, mf.mo_occ, result), jax)
        facade_seconds = time.perf_counter() - started
        if profile:
            profile.disable()
        call_timer.cancel()
        event("facade_end", seconds=facade_seconds, ru_maxrss_bytes=peak_rss_bytes())
        if profile:
            # Persist SCF evidence before trace export or kernels can time out.
            profile.dump_stats(str(args.output / "scf.prof"))
            with (args.output / "cprofile.txt").open("w") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(100)
        if stages:
            stages.restore()
            save("stage_summary.json", stages.summary())
            inputs = stages.captured.get("inputs")
            result = stages.captured.get("result", result)
            shared = stages.captured.get("shared_result")
            if shared is not None:
                history = shared.history
                save("scf_history.json", {name: np.asarray(getattr(history, name)).tolist()
                    for name in ("density_rms", "energy_change", "gradient", "converged")})
        else:
            inputs = getattr(mf, "_scf_inputs", None)
        if hasattr(result, "orbital_gradient_norm"):
            gradient = float(result.orbital_gradient_norm)
        else:
            state = mf.scf_result if mf.scf_result is not None else mf.reference
            gradient = float(orbital_gradient_norm(state.fock_matrix, mf.mo_coeff, mf.mo_occ))
        finite = bool(np.isfinite(float(mf.e_tot)) and np.isfinite(gradient)
                      and np.isfinite(np.asarray(mf.mo_coeff)).all())
        summary = dict(mode=args.mode, case_id=cfg["case_id"], method=cfg["method"],
            xc=cfg["xc"], basis=cfg["basis"], representation=representation,
            nao=int(nao), mo_coeff_shape=list(mf.mo_coeff.shape), dtype=str(mf.mo_coeff.dtype),
            cycles=int(mf.cycles), converged=bool(mf.converged), energy_hartree=float(mf.e_tot),
            fixed_cycle_probe=args.fixed_cycles is not None, snapshot_converged=bool(mf.converged),
            gradient_norm=gradient, gradient_tolerance=cfg["gradient_tol"],
            scientific_pass=bool(args.fixed_cycles is None and mf.converged and finite
                                 and gradient < cfg["gradient_tol"]),
            status=("fixed_cycle_probe" if args.fixed_cycles is not None else
                    "ok" if mf.converged and finite else "nonconverged"),
            scf_facade_seconds=facade_seconds, ru_maxrss_bytes=peak_rss_bytes())
        if inputs is not None:
            save("input_arrays.json", array_metadata(inputs))
        save("summary.json", summary)
        event("scf_certificate", **summary)
        if trace_active:
            jax.profiler.stop_trace()
            trace_active = False
            event("jax_trace_stopped")
        if stages and inputs is not None:
            remaining = max(0., deadline - time.perf_counter() - 5.)
            if remaining >= 5.:
                # Import late: microbenchmark dependencies cannot affect SCF timing.
                event("kernel_import_begin")
                from profile_kernels import run_kernels
                remaining = max(0., deadline - time.perf_counter() - 5.)
                if remaining >= 5.:
                    event("kernels_begin", budget_seconds=remaining)
                    run_kernels(inputs, result, cfg, args.output / "kernels.jsonl", remaining)
                    event("kernels_end", ru_maxrss_bytes=peak_rss_bytes())
                else:
                    event("kernels_skipped", reason="Kernel import exhausted remaining budget")
            else:
                event("kernels_skipped", reason="Less than five seconds remain after cleanup reserve")
        elif stages:
            event("kernels_skipped", reason="No prepared inputs captured")
        summary.update(process_seconds=time.perf_counter() - process_start,
                       ru_maxrss_after_kernels_bytes=peak_rss_bytes())
        save("summary.json", summary)
        event("process_end", status="ok", ru_maxrss_bytes=peak_rss_bytes())
        return 0
    except Exception as error:
        traceback.print_exc()
        summary = dict(status="error", error=f"{type(error).__name__}: {error}",
                       process_seconds=time.perf_counter() - process_start, ru_maxrss_bytes=peak_rss_bytes())
        save("failure.json", summary)
        event("process_error", **summary)
        return 1
    finally:
        if call_timer:
            call_timer.cancel()
        if stages:
            stages.restore()
            save("stage_summary.json", stages.summary())
        if profile and not (args.output / "scf.prof").exists():
            profile.disable()
            profile.dump_stats(str(args.output / "scf.prof"))
            with (args.output / "cprofile.txt").open("w") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(100)
        if trace_active:
            try:
                jax.profiler.stop_trace()
            except Exception as error:
                event("jax_trace_stop_error", error=f"{type(error).__name__}: {error}")
        runtime.update(process_seconds=time.perf_counter() - process_start, ru_maxrss_bytes=peak_rss_bytes())
        save("runtime.json", runtime)
        timer.cancel()
        events.flush()


if __name__ == "__main__":
    raise SystemExit(main())
