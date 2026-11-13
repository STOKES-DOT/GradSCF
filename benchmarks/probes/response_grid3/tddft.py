"""Independent DFT -> TDA/TDDFT cases; run only inside a Slurm allocation.

Usage: PYTHONPATH=src python benchmarks/probes/response_grid3/tddft.py case.json out
Required JSON: atom. Defaults: water-compatible closed-shell RKS/PBE, Cartesian
sto-3g, grid 3, three roots, CPU, exact full integrals, cold + 3 warm new objects.
Controls: ks, xc, basis, grid_level, full, representation (full/direct/df),
auxbasis, device, threads, max_memory (decimal MB), grid_block_size,
response_grid_block_size, repeats, td_tol, operator_ad. No numerical kernel is
implemented here. Certification/optional vector AD run after the clean marker.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]


def configuration(data):
    cfg = dict(case_id="response-grid3", variant="baseline", engine="gradscf",
        ks="rks", xc="pbe", basis="sto-3g", unit="Angstrom", charge=0, spin=0,
        grid_level=3, full=False, roots=3, device="cpu", threads=1, blas_threads=1,
        representation="full", auxbasis="weigend", max_memory=60000.,
        grid_block_size=0, response_grid_block_size=8192,
        scf_tol=1e-12, density_tol=1e-11, gradient_tol=1e-10,
        td_tol=1e-8, max_cycle=100, td_max_cycle=200, repeats=4,
        direct_tol=0., integral_cache="auto", operator_ad=False)
    cfg.update(data)
    if not cfg.get("atom"):
        raise ValueError("An explicit atom specification is required")
    if cfg["engine"] != "gradscf":
        raise ValueError("This runner measures GradSCF; reference engines use separate processes")
    for key, choices in (("ks", ("rks", "uks", "gks")),
                         ("device", ("cpu", "gpu")),
                         ("representation", ("full", "direct", "df"))):
        if cfg[key] not in choices:
            raise ValueError(f"Invalid {key}: {cfg[key]!r}")
    for key in ("roots", "threads", "blas_threads", "max_cycle", "td_max_cycle", "repeats"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    return cfg


def amplitudes(result, full, np):
    if hasattr(result, "amplitudes"):
        names = ("amplitudes",)
    elif hasattr(result, "x_amplitudes"):
        names = ("x_amplitudes", "y_amplitudes")
    elif full:
        names = ("x_amplitudes_alpha", "x_amplitudes_beta",
                 "y_amplitudes_alpha", "y_amplitudes_beta")
    else:
        names = ("amplitudes_alpha", "amplitudes_beta")
    nroots = len(result.excitation_energies)
    return np.concatenate([np.asarray(getattr(result, name)).reshape(nroots, -1)
                           for name in names], axis=1), names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cfg = configuration(json.loads(args.config.read_text()))
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Numerical runs require a remote Slurm allocation")
    allowed = sorted(os.sched_getaffinity(0))
    if cfg["threads"] > len(allowed):
        parser.error("Requested threads exceed the Slurm CPU affinity")
    os.sched_setaffinity(0, allowed[:cfg["threads"]])
    out = args.output
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("Use a new or empty output directory; previous raw records are immutable")
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(cfg["blas_threads"])
    os.environ["GRADSCF_NATIVE_THREADS"] = str(cfg["threads"])
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    from gradscf import dft, gto, tdscf
    from gradscf.integrals._native import register_integrals

    cpu, target = jax.devices("cpu")[0], jax.devices(cfg["device"])[0]
    jax.config.update("jax_default_device", cpu)
    spec = importlib.util.spec_from_file_location(
        "response_grid3_helpers", ROOT / "benchmarks/probes/tddft_perf/worker.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    ready, plain = helpers.ready, helpers.plain
    log = helpers.Recorder(cfg, out)
    from benchmarks.probes.tddft_execution.worker import compile_counter, workload_counter
    if cfg.get("compile_diagnostic"):
        jax.config.update("jax_log_compiles", True)
        jax.config.update("jax_explain_cache_misses", True)
    if cfg.get("solver_diagnostic"):
        workload_counter(out)

    def record_json(name, value):
        (out / name).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")

    def active(stage, repeat):
        state = dict(stage=stage, repeat=repeat, started_monotonic=time.monotonic(),
            process_budget_seconds=cfg.get("process_budget_seconds", 6000))
        record_json("active_stage.json", state)
        print(json.dumps(state), flush=True)
        if cfg["device"] == "gpu":
            stats = target.memory_stats() or {}
            log.add("device_memory_snapshot", None, repeat, active_stage=stage,
                gpu_memory_bytes=stats.get("bytes_in_use"),
                gpu_memory_metric_kind="JAX allocator snapshot; not a sampled process peak",
                gpu_peak_bytes_in_use=stats.get("peak_bytes_in_use"),
                gpu_pool_bytes=stats.get("pool_bytes"))

    def finite(*values):
        return all(np.isfinite(np.asarray(value)).all() for value in values)

    if cfg.get("solver_diagnostic"):
        from gradscf.tddft import response
        original_builder = response._build_restricted_response_operator_data
        def observed_builder(*args, **kwargs):
            data = original_builder(*args, **kwargs)
            with (out / "kernel_plans.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(data.memory_plan._asdict())) + "\n")
            return data
        response._build_restricted_response_operator_data = observed_builder

    source_root = Path(gradscf.__file__).resolve().parent
    sources = ("scf/facade.py", "scf/gks.py", "tdscf/api.py", "tddft/execution.py",
               "tddft/response.py", "tddft/unrestricted.py", "tddft/casida.py",
               "tddft/_semilocal_response.py", "tddft/_unrestricted_semilocal_response.py",
               "tddft/response_options.py")
    native = Path(register_integrals()._name).resolve()
    record_json("runtime.json", dict(python=sys.executable, source=gradscf.__file__,
        source_files={name: dict(path=str(source_root / name),
            sha256=hashlib.sha256((source_root / name).read_bytes()).hexdigest()) for name in sources},
        input_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        native_path=str(native), native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        slurm_job_id=os.environ["SLURM_JOB_ID"], affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "GRADSCF_NATIVE_THREADS")},
        versions=dict(jax=jax.__version__, numpy=np.__version__),
        devices=[dict(name=str(d), kind=d.device_kind) for d in jax.devices()],
        units="Ha, Angstrom input, seconds, bytes; max_memory in decimal MB",
        cache_environment={key: os.environ.get(key) for key in
            ("JAX_ENABLE_COMPILATION_CACHE", "JAX_COMPILATION_CACHE_DIR")}))
    td = ref = None
    repeat = 0
    try:
        for repeat in range(cfg["repeats"]):
            active("end_to_end", repeat)
            start = time.perf_counter()
            mol = log.timed("molecule", lambda: gto.M(atom=cfg["atom"], basis=cfg["basis"],
                unit=cfg["unit"], charge=cfg["charge"], spin=cfg["spin"], cart=True), repeat)
            mf = getattr(dft, cfg["ks"].upper())(mol, xc=cfg["xc"],
                grids_level=cfg["grid_level"], execution_device=cfg["device"],
                conv_tol=cfg["scf_tol"], conv_tol_density=cfg["density_tol"],
                conv_tol_grad=cfg["gradient_tol"], max_cycle=cfg["max_cycle"],
                max_memory=cfg["max_memory"], grid_block_size=cfg["grid_block_size"],
                init_guess=cfg.get("init_guess", "hcore"), grid_screen_tol=cfg.get("grid_screen_tol", 0.))
            if cfg["representation"] == "direct":
                mf.direct_scf_workers, mf.direct_scf_tol = cfg["threads"], cfg["direct_tol"]
                mf.direct_scf()
            elif cfg["representation"] == "df":
                if cfg["ks"] == "gks":
                    mf.density_fit()  # Preserve the facade's explicit unsupported error.
                else:
                    mf.density_fit(cfg["auxbasis"])
            active("scf", repeat)
            log.timed("scf", lambda: mf.run().e_tot, repeat)
            passed = bool(mf.converged) and finite(mf.e_tot, mf.mo_coeff, mf.mo_energy)
            log.add("scf_status", None, repeat, scientific_pass=passed,
                scf_energy=float(mf.e_tot), scf_iterations=int(mf.cycles), ks=cfg["ks"])
            if not passed:
                raise ArithmeticError("SCF failed convergence/finite checks")
            if cfg.get("save_scf_state", False):
                # Recovery evidence only: no converged density is injected into a timed run.
                np.savez(out / "scf_state.npz", mo_coeff=np.asarray(mf.mo_coeff),
                    mo_occ=np.asarray(mf.mo_occ), mo_energy=np.asarray(mf.mo_energy),
                    energy=np.asarray(mf.e_tot))
            cls = tdscf.TDDFT if cfg["full"] else tdscf.TDA
            td = cls(mf, nstates=cfg["roots"], eigensolver="davidson",
                davidson_tol=cfg["td_tol"], davidson_max_iter=cfg["td_max_cycle"],
                response_kernel_options=dict(
                    two_electron_mode=cfg["representation"] if cfg["representation"] != "full" else "auto",
                    max_memory=cfg["max_memory"], direct_workers=cfg["threads"],
                    direct_tol=cfg["direct_tol"], integral_cache=cfg["integral_cache"],
                    grid_block_size=cfg["response_grid_block_size"]))
            active("response_reference", repeat)
            ref = log.timed("response_reference", lambda: td.molecule, repeat)
            occ = np.asarray(mf.mo_occ)
            counts = (np.count_nonzero(occ, axis=1).tolist() if cfg["ks"] == "uks"
                      else int(np.count_nonzero(occ)))
            nmo = int(mf.mo_coeff.shape[-1])
            log.context.update(nao=int(mf.mo_coeff.shape[-2]), nocc=counts,
                nvir=[nmo - n for n in counts] if isinstance(counts, list) else nmo - counts,
                ngrid=int(ref.grid.weights.size),
                naux=None if ref.df_factors is None else int(ref.df_factors.shape[0]))
            if cfg["ks"] == "rks":
                dim = counts * (nmo - counts)
                log.add("dimensions", None, repeat, ntransition=dim,
                    hypothetical_dense_a_bytes=8 * dim**2,
                    hypothetical_full_ao_eri_bytes=8 * log.context["nao"]**4,
                    notes="Size estimates only; these tensors are not requested or allocated")
            active("response_solve", repeat)
            with jax.default_device(target):
                restore = compile_counter(out, repeat) if cfg.get("compile_diagnostic") else lambda: None
                try:
                    log.timed("response_solve", td.kernel, repeat)
                finally:
                    restore()
                strengths = log.timed("oscillator_strength", td.oscillator_strength, repeat)
            passed = bool(np.all(td.converged)) and finite(td.e, strengths)
            log.add("end_to_end", time.perf_counter() - start, repeat,
                public_converged=bool(np.all(td.converged)), finite_outputs=finite(td.e, strengths),
                scientific_pass=None, excitations_ha=td.e, oscillator_strength=strengths,
                cache_state="cold process" if repeat == 0 else "new objects; fixed geometry",
                timing_scope="independent geometry -> SCF -> response -> oscillator strengths",
                notes="Original-operator certification follows clean phase; no restart vectors")
            if not passed:
                raise ArithmeticError("TD response failed convergence/finite checks")
            if repeat + 1 < cfg["repeats"]:
                td = ref = None
                del mf, mol, occ, strengths
                gc.collect()  # Release object cycles while retaining compilation caches.
        record_json("clean_phase_done.json", dict(ended_monotonic=time.monotonic(),
            ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            metric="Linux process cumulative high-water mark before certification"))
        active("original_operator_certificate", repeat)
        vectors, names = amplitudes(td.result, cfg["full"], np)
        with jax.default_device(target):
            vind = log.timed("certificate_preparation", td._solver.gen_tdhf_vind
                             if cfg["full"] else td._solver.gen_tda_vind)
            z = jax.device_put(vectors, target)
            acted = np.asarray(log.timed("certificate_action", lambda: vind(z)))
        residual = np.linalg.norm(acted - np.asarray(td.e)[:, None] * vectors, axis=1)
        residual /= np.linalg.norm(vectors, axis=1)
        passed = finite(residual) and bool(np.all(residual <= cfg["td_tol"]))
        log.add("certification", None, repeat, residual_per_root=residual,
            original_operator_residual=float(np.max(residual)), scientific_pass=passed,
            notes="norm(Lz-omega*z)/norm(z), original src operator, final clean repeat")
        payload = {name: np.asarray(getattr(td.result, name)) for name in names}
        payload.update(energies=np.asarray(td.e), mo_coeff=np.asarray(ref.mo_coeff),
            mo_occ=np.asarray(ref.mo_occ), mo_energy=np.asarray(ref.mo_energy),
            overlap=np.asarray(ref.overlap_matrix))
        np.savez(out / "states.npz", **payload)
        if cfg["operator_ad"]:
            active("vector_ad", repeat)
            # Explicitly an amplitude derivative; does not certify geometry or XC parameter AD.
            direction = jax.device_put(np.random.default_rng(7).normal(size=z.shape), target)
            cotangent = jax.device_put(np.random.default_rng(11).normal(size=z.shape), target)
            with jax.default_device(target):
                jvp = log.timed("vector_jvp", lambda: jax.jvp(vind, (z,), (direction,))[1])
                vjp = log.timed("vector_vjp", lambda: jax.vjp(vind, z)[1](cotangent)[0])
            lhs, rhs = jnp.vdot(cotangent, jvp), jnp.vdot(vjp, direction)
            error = float(jnp.abs(lhs - rhs)) / max(1., float(jnp.abs(lhs)), float(jnp.abs(rhs)))
            ad_pass = finite(jvp, vjp) and error <= 1e-9
            log.add("vector_ad", None, scientific_pass=ad_pass, duality_relative_error=error,
                notes="JVP/VJP wrt response-vector input only; not a geometry/parameter AD gate")
            passed = passed and ad_pass
        record_json("status.json", dict(status="ok" if passed else "failed", scientific_pass=passed))
        return 0 if passed else 1
    except Exception as exc:
        status = "unsupported" if isinstance(exc, NotImplementedError) else "error"
        log.add("failure", None, repeat, status=status, scientific_pass=False,
                exception_type=type(exc).__name__, message=str(exc))
        record_json("status.json", dict(status=status, scientific_pass=False,
            exception_type=type(exc).__name__, message=str(exc)))
        traceback.print_exc()
        return 2 if status == "unsupported" else 1


if __name__ == "__main__":
    sys.exit(main())
