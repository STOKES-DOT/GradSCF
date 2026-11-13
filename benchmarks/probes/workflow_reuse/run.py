"""Bounded public SCF -> TD and DFT -> GW -> BSE workflows, inside Slurm only.

This module orchestrates src/gradscf APIs and records synchronized complete
workflows. It implements no electronic-structure equations. Fresh objects are
used for every repeat; the first case includes process-cold JAX compilation.
Later first-case calls may share programs compiled by preceding cases.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


MODES = ("rhf", "uhf", "ghf", "rks", "uks", "gks")
METHODS = ("g0w0", "evgw", "qsgw", "scgw")
ATOMS = {
    "h2": "H 0 0 0; H 0 0 .74",
    "water": "O 0 0 0; H 0 -.757 .587; H 0 .757 .587",
    "benzene": ("C 1.397 0 0; C .6985 1.209837489 0; C -.6985 1.209837489 0; "
                "C -1.397 0 0; C -.6985 -1.209837489 0; C .6985 -1.209837489 0; "
                "H 2.481 0 0; H 1.2405 2.148609027 0; H -1.2405 2.148609027 0; "
                "H -2.481 0 0; H -1.2405 -2.148609027 0; H 1.2405 -2.148609027 0"),
}


def mean_field(mode, *, molecule="h2", basis="sto-3g", xc="pbe", grid=0,
               max_memory=4096., backend="full", threads=1):
    """Shared real molecular preparation for the integration tests and probe."""
    from gradscf import dft, gto, scf

    mol = gto.M(atom=ATOMS[molecule], basis=basis, unit="Angstrom", cart=True)
    cls = getattr(scf if mode.endswith("hf") else dft, mode.upper())
    controls = dict(conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9,
                    max_cycle=100, grids_level=grid, max_memory=max_memory,
                    execution_device="cpu", jk_backend=backend,
                    direct_scf_workers=threads, direct_scf_tol=0.)
    if mode.endswith("ks"):
        controls["xc"] = xc
    return cls(mol, **controls)


def quasiparticles(source, method, *, nw=16, max_memory=4096.):
    from gradscf import gw

    return gw.GW(source, method=method, nw=nw, max_cycle=80, conv_tol=1e-7,
                 damp=.3, beta=20., mixing=.5, max_memory=max_memory)


def td(source, full):
    factory = source.TDDFT if full else source.TDA
    return factory(nstates=1, eigensolver="davidson", davidson_tol=1e-9,
                   davidson_max_iter=80)


def excitons(source, full):
    from gradscf import bse

    return bse.BSE(source, nroots=1, tda=not full, solver="davidson",
                   conv_tol=1e-9, max_cycle=100)


def _safe(value):
    """Keep raw records strict JSON without converting complex physics to real."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if isinstance(value, complex):
        return dict(real=_safe(value.real), imag=_safe(value.imag))
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return None
    return value


def _rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else 1024 * peak)


def _manifest(args, jax, gradscf):
    from gradscf.integrals._native import register_integrals

    package = Path(gradscf.__file__).resolve().parent
    library = Path(register_integrals()._name).resolve()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package,
                              text=True, capture_output=True, check=False)
    paths = sorted(package.rglob("*.py"))
    return dict(arguments={k: str(v) if isinstance(v, Path) else v
                           for k, v in vars(args).items()},
                source=str(package), git_head=os.getenv("GRADSCF_SOURCE_COMMIT") or
                (revision.stdout.strip() if revision.returncode == 0 else None),
                source_sha256={str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths},
                probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                native_path=str(library), native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                python=sys.executable, platform=platform.platform(), processor=platform.processor(),
                versions={name: importlib.metadata.version(name)
                          for name in ("jax", "jaxlib", "numpy", "scipy")},
                devices=[str(device) for device in jax.devices()],
                affinity=sorted(os.sched_getaffinity(0)),
                slurm={key: os.getenv(key) for key in
                       ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURMD_NODENAME")},
                environment={key: os.getenv(key) for key in
                             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                              "GRADSCF_NATIVE_THREADS", "JAX_ENABLE_COMPILATION_CACHE")},
                molecule=dict(atom=ATOMS[args.molecule], basis=args.basis, unit="Angstrom",
                              charge=0, spin=0, cart=True),
                tolerances=dict(scf_energy_hartree=1e-11, scf_density=1e-10,
                                scf_orbital_gradient=1e-9, td_residual=1e-9,
                                gw_outer_hartree=1e-7, bse_residual_hartree=1e-9),
                timing_scope="fresh public objects, complete synchronized workflow; no supplied QP poles",
                memory_scope="cumulative process peak RSS; not per-phase incremental memory",
                integral_representation=("native packed four-center SCF; GW exact-pair spectral factors tol=1e-12"
                                         if args.backend == "full" else
                                         "native direct SCF; independent auxiliary RI (weigend) for GW"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--suite", choices=("scf-td", "gw-bse"), required=True)
    parser.add_argument("--molecule", choices=tuple(ATOMS), default="h2")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid", type=int, default=0)
    parser.add_argument("--backend", choices=("full", "direct"), default="full")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--nw", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-memory", type=float, default=4096.)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SLURM_STEP_ID"):
        parser.error("Numerical execution requires a Slurm job step; use srun.")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("Use a new output path; raw records are immutable.")
    if args.repeats < 0 or args.nw < 1 or args.threads < 1:
        parser.error("repeats must be nonnegative; nw and threads must be positive.")
    allowed = sorted(os.sched_getaffinity(0))
    if args.threads > len(allowed):
        parser.error("Requested CPUs exceed the Slurm affinity.")
    os.sched_setaffinity(0, allowed[:args.threads])
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import numpy as np
    import gradscf

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(_manifest(args, jax, gradscf), indent=2)+"\n")

    def record(**row):
        common = dict(suite=args.suite, molecule=args.molecule, basis=args.basis,
                      backend=args.backend, xc=args.xc, grid=args.grid, dtype="float64",
                      units="Ha, Angstrom, s, bytes", peak_rss_bytes=_rss_bytes())
        with args.output.open("a") as stream:
            stream.write(json.dumps(_safe(dict(common, **row)), allow_nan=False)+"\n")

    failures = 0
    names = args.modes if args.suite == "scf-td" else args.methods
    for name in names:
        for repeat in range(args.repeats+1):
            context = dict(case=name, repeat=repeat,
                           cache_state="first_case_call" if repeat == 0 else "warm_fresh_objects")
            started = time.perf_counter()
            stage = "scf"
            try:
                source = mean_field(name if args.suite == "scf-td" else "rks",
                                    molecule=args.molecule, basis=args.basis, xc=args.xc,
                                    grid=args.grid, max_memory=args.max_memory,
                                    backend=args.backend, threads=args.threads)
                tick = time.perf_counter()
                source.run()
                jax.block_until_ready((source.e_tot, source.mo_energy, source.mo_coeff))
                record(**context, stage=stage, seconds=time.perf_counter()-tick,
                       energy_hartree=float(source.e_tot), cycles=source.cycles,
                       mo_coeff_shape=tuple(source.mo_coeff.shape),
                       converged=bool(source.converged), scientific_pass=bool(source.converged))
                if not source.converged:
                    raise ArithmeticError("SCF did not converge")
                ground_result = source.scf_result
                if args.suite == "gw-bse":
                    stage = "gw"
                    calculation = quasiparticles(source, name, nw=args.nw, max_memory=args.max_memory)
                    if args.backend == "direct":
                        calculation.density_fit("weigend")
                    tick = time.perf_counter()
                    calculation.run()
                    jax.block_until_ready(calculation.result)
                    result = calculation.result
                    details = {key: getattr(result, key, None) for key in
                               ("mo_energy", "screening_energy", "qp_residual", "qp_computed_mask",
                                "converged_mask", "qp_branch_valid", "n_iter", "fixed_point_residual",
                                "particle_number_error", "total_energy")}
                    finite = all(np.isfinite(np.asarray(getattr(result, key))).all() for key in
                                 (("self_energy_iw", "fixed_point_residual", "particle_number_error")
                                  if name == "scgw" else ("mo_energy", "screening_energy")))
                    valid = bool(calculation.converged) and finite
                    if name != "scgw":
                        valid &= bool(np.asarray(result.qp_computed_mask).all())
                        valid &= bool(np.asarray(result.converged_mask).all())
                        if result.qp_branch_valid is not None:
                            valid &= bool(np.asarray(result.qp_branch_valid).all())
                        if result.qp_residual is not None:
                            valid &= bool(np.isfinite(np.asarray(result.qp_residual)).all())
                    record(**context, stage=stage, seconds=time.perf_counter()-tick,
                           converged=bool(calculation.converged), scientific_pass=valid, **details)
                    if not valid:
                        raise ArithmeticError("GW convergence, coverage or finiteness failed")
                else:
                    calculation = source
                for full in (False, True):
                    stage = ("bse_" if args.suite == "gw-bse" else "td_") + ("full" if full else "tda")
                    expected_rejection = name == "scgw" if args.suite == "gw-bse" else name in ("ghf", "gks")
                    tick = time.perf_counter()
                    try:
                        response = excitons(calculation, full) if args.suite == "gw-bse" else td(calculation, full)
                        response.run() if args.suite == "gw-bse" else response.kernel()
                        jax.block_until_ready(response.result)
                    except NotImplementedError as error:
                        if not expected_rejection:
                            raise
                        record(**context, stage=stage, seconds=time.perf_counter()-tick,
                               status="verified_unsupported", reason=str(error), scientific_pass=False,
                               capability_guard_pass=True)
                        continue
                    if expected_rejection:
                        raise AssertionError("Capability changed: update and scientifically validate this probe")
                    energies = np.asarray(response.e)
                    details = {key: getattr(response.result, key, None) for key in
                               ("residual_norms", "stable", "response_valid", "screening_valid")}
                    valid = bool(np.asarray(response.converged).all()) and bool(np.isfinite(energies).all())
                    if args.suite == "gw-bse":
                        valid &= bool(np.asarray(response.result.stable).all())
                        valid &= bool(np.asarray(response.result.screening_valid).all())
                        valid &= bool(np.isfinite(np.asarray(response.result.residual_norms)).all())
                    record(**context, stage=stage, seconds=time.perf_counter()-tick,
                           excitation_hartree=energies, converged=bool(np.asarray(response.converged).all()),
                           scientific_pass=valid, **details)
                    if not valid:
                        raise ArithmeticError("Response did not converge to finite stable roots")
                if source.scf_result is not ground_result:
                    raise AssertionError("A downstream consumer replaced the completed SCF result")
                record(**context, stage="workflow", seconds=time.perf_counter()-started,
                       status="complete", scientific_pass=name not in ("ghf", "gks", "scgw"),
                       capability_guard_pass=name in ("ghf", "gks", "scgw"))
            except Exception as error:
                failures += 1
                record(**context, stage=stage, status="error", error_type=type(error).__name__,
                       reason=str(error), seconds=time.perf_counter()-started, scientific_pass=False)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
