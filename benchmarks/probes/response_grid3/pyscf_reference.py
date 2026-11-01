"""Independent PySCF DFT -> TDA/TDDFT reference; remote Slurm execution only.

Usage: python pyscf_reference.py case.json out
Accepts tddft.py JSON keys, creates cold + three warm fresh objects, and imports
no JAX or GradSCF physics. Shared Recorder is an I/O-only benchmark utility.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]


def configuration(data):
    cfg = dict(case_id="response-grid3", variant="reference", ks="rks", xc="pbe",
        basis="sto-3g", unit="Angstrom", charge=0, spin=0, grid_level=3,
        full=False, roots=3, device="cpu", threads=1, blas_threads=1,
        representation="full", auxbasis="weigend", max_memory=60000.,
        scf_tol=1e-12, density_tol=1e-11, gradient_tol=1e-10, td_tol=1e-8,
        max_cycle=100, td_max_cycle=200, repeats=4, direct_tol=0., cart=True)
    cfg.update(data)
    cfg["engine"] = "pyscf"
    if not cfg.get("atom"):
        raise ValueError("An explicit atom specification is required")
    for key, choices in (("ks", ("rks", "uks", "gks")),
                         ("representation", ("full", "direct", "df"))):
        if cfg[key] not in choices:
            raise ValueError(f"Invalid {key}: {cfg[key]!r}")
    for key in ("roots", "threads", "blas_threads", "max_cycle", "td_max_cycle"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["repeats"] != 4 or cfg["grid_level"] != 3 or cfg.get("grids_level", 3) != 3:
        raise ValueError("Reference protocol requires grid3 and exactly four fresh calls")
    if cfg["cart"] is not True or not isinstance(cfg["full"], bool):
        raise ValueError("Cartesian basis and an explicit boolean full/TDA choice are required")
    if cfg["scf_tol"] != 1e-12 or cfg["gradient_tol"] != 1e-10:
        raise ValueError("Reference SCF thresholds are fixed at 1e-12 Ha and 1e-10 orbital gradient")
    if cfg.get("frozen_core", 0) != 0 or cfg.get("orbs") is not None:
        raise ValueError("The reference retains every orbital; no frozen-core/target window")
    for key in ("occupied_window", "virtual_window"):
        if cfg.get(key, "all") != "all":
            raise ValueError("The reference retains the complete occupied/virtual space")
    return cfg


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
        parser.error("Use a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.json").write_text(json.dumps(cfg, indent=2) + "\n")
    reason = None
    if cfg["ks"] == "gks":
        reason = "Not benchmarked because GradSCF whole-spinor response is unavailable; PySCF GKS response capability has not been tested here."
    elif cfg["device"] != "cpu":
        reason = "This independent reference runner is CPU-only; no GPU reference capability is inferred."
    if reason:
        (out / "status.json").write_text(json.dumps(dict(status="unsupported",
            scientific_pass=False, engine="pyscf", message=reason,
            probe_sha256=digest(__file__)), indent=2) + "\n")
        return 2
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cfg["blas_threads"])
    import numpy as np
    import scipy
    import pyscf
    from pyscf import dft, gto, lib, tdscf
    lib.num_threads(cfg["blas_threads"])
    helper_path = ROOT / "benchmarks/probes/tddft_perf/worker.py"
    spec = importlib.util.spec_from_file_location("response_grid3_io", helper_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    log = helpers.Recorder(cfg, out)

    def record(name, value):
        (out / name).write_text(json.dumps(helpers.plain(value), indent=2, allow_nan=False) + "\n")

    def active(stage, repeat):
        record("active_stage.json", dict(stage=stage, repeat=repeat,
            started_monotonic=time.monotonic(), solve_budget_seconds=6000))

    def finite(*values):
        return all(np.isfinite(np.asarray(v)).all() for v in values)

    def array_hash(value):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    original_wb97xd = str(cfg["xc"]).lower().replace("-", "").replace("_", "") in (
        "wb97xd", "hybggaxcwb97xd")
    xc = "HYB_GGA_XC_WB97X_D" if original_wb97xd else cfg["xc"]
    source_note = dict(
        scope="Independent PySCF electronic SCF and TDA/full-TDDFT, complete RKS or UKS same-spin transition space.",
        gks="Not benchmarked because GradSCF whole-spinor response is unavailable; no inference about PySCF capability.",
        wB97X_D=dict(requested=original_wb97xd, actual_xc=xc,
            chg_dispersion_included=False if original_wb97xd else None,
            total_energy_comparable=False if original_wb97xd else None,
            note="Original LibXC wB97X-D electronic functional only; no CHG pair correction, and no substitution by wB97X/D3/D4. Compare fixed-geometry electronic response only after numerical checks; full total-energy equivalence is not claimed."),
        scf="PySCF conv_tol=1e-12 and conv_tol_grad=1e-10. GradSCF density_tol is preserved as input metadata but is not an independent PySCF convergence criterion.",
        timing="Imports excluded; each timed call builds fresh molecule/SCF/TD objects from hcore; no previous density or response vectors reused. Cold means first solve in this process, not cold filesystem caches.",
        representation="full prebuilds ordinary Coulomb s8 ERIs; direct requests PySCF direct_scf but PySCF may retain in-core ERIs when memory permits; actual storage is logged. df uses the explicit auxiliary basis. Range-separated exchange follows PySCF's own implementation.",
        docs=["https://pyscf.org/user/tddft.html", "https://pyscf.org/_modules/pyscf/scf/hf.html",
              "https://pyscf.org/_modules/pyscf/scf/dispersion.html", "https://pyscf.org/_modules/pyscf/dft/libxc.html"])
    record("source_note.json", source_note)
    modules = ["pyscf", "pyscf.gto.mole", "pyscf.dft.rks", "pyscf.dft.uks",
               "pyscf.dft.numint", "pyscf.dft.gen_grid", "pyscf.dft.libxc",
               "pyscf.scf.hf", "pyscf.scf.uhf", "pyscf.tdscf.rks", "pyscf.tdscf.uks",
               "pyscf.tdscf.rhf", "pyscf.tdscf.uhf", "pyscf.df.df_jk"]
    paths = {name: Path(importlib.import_module(name).__file__).resolve() for name in modules}
    record("runtime.json", dict(python=sys.executable, platform=platform.platform(),
        versions=dict(python=platform.python_version(), pyscf=pyscf.__version__, numpy=np.__version__,
                      scipy=scipy.__version__, libxc=dft.libxc.__version__),
        source_files={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()},
        probe_sha256=digest(__file__), helper_path=str(helper_path), helper_sha256=digest(helper_path),
        input_file_sha256=digest(args.config), normalized_input_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        slurm_job_id=os.environ["SLURM_JOB_ID"], affinity=sorted(os.sched_getaffinity(0)),
        pyscf_openmp_threads=lib.num_threads(), thread_environment={name: os.getenv(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        implementation_flags=dict(jax_imported="jax" in sys.modules, gradscf_imported="gradscf" in sys.modules,
            cart=True, small_rho_cutoff=0, init_guess="hcore", frozen_core=0, singlet=cfg["ks"] == "rks"),
        units="Ha, Angstrom input unless unit says otherwise, seconds, bytes; max_memory in decimal MB"))
    repeat = 0
    try:
        for repeat in range(4):
            active("end_to_end", repeat)
            start = time.perf_counter()
            mol = log.timed("molecule", lambda: gto.M(atom=cfg["atom"], basis=cfg["basis"],
                unit=cfg["unit"], charge=cfg["charge"], spin=cfg["spin"], cart=True,
                verbose=4, output=str(out / f"pyscf-{repeat}.log")), repeat)
            # The explicit representation chooses UKS even when spin == 0.
            mf = (dft.UKS if cfg["ks"] == "uks" else dft.RKS)(mol)
            mf.xc, mf.grids.level = xc, 3
            mf._numint.libxc = dft.libxc
            mf.small_rho_cutoff, mf.init_guess = 0, "hcore"
            mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
            mf.max_cycle, mf.max_memory = cfg["max_cycle"], cfg["max_memory"]
            mf.direct_scf_tol = cfg["direct_tol"]
            mf.disp = None
            if cfg["representation"] == "df":
                mf = mf.density_fit(auxbasis=cfg["auxbasis"])
            elif cfg["representation"] == "full":
                mf.direct_scf = False
                mf._eri = log.timed("integral_preparation", lambda: mol.intor("int2e", aosym="s8"), repeat)
            else:
                mf.direct_scf = True
            history = []
            def callback(env):
                history.append({key: float(env[key]) for key in
                    ("cycle", "e_tot", "norm_gorb", "norm_ddm") if key in env})
            mf.callback = callback
            active("scf", repeat)
            log.timed("scf", mf.kernel, repeat)
            occ = np.asarray(mf.mo_occ)
            nmo = int(np.asarray(mf.mo_coeff).shape[-1])
            nocc = np.count_nonzero(occ, axis=1).tolist() if cfg["ks"] == "uks" else int(np.count_nonzero(occ))
            nvir = [nmo - n for n in nocc] if isinstance(nocc, list) else nmo - nocc
            dim = sum(n*v for n, v in zip(nocc, nvir)) if isinstance(nocc, list) else nocc*nvir
            log.context.update(nao=int(mol.nao_nr()), nocc=nocc, nvir=nvir,
                ngrid=len(mf.grids.weights), naux=mf.with_df.get_naoaux() if cfg["representation"] == "df" else None)
            scf_pass = bool(mf.converged) and finite(mf.e_tot, mf.mo_energy, mf.mo_coeff, occ)
            log.add("scf_status", None, repeat, scientific_pass=scf_pass,
                scf_energy=float(mf.e_tot), converged=bool(mf.converged), scf_iterations=getattr(mf, "cycles", None),
                callback_history=history, nmo=nmo, transition_dimension=dim,
                electron_count=float(occ.sum()), mo_energy_ha=mf.mo_energy, mo_occ=occ,
                spin_square=mf.spin_square(), dispersion_summary=mf.scf_summary.get("dispersion"),
                chg_total_energy_comparable=not original_wb97xd,
                actual_eri_shape=None if mf._eri is None else list(mf._eri.shape))
            if not scf_pass:
                raise ArithmeticError("PySCF SCF failed convergence/finite checks")
            td = tdscf.TDDFT(mf) if cfg["full"] else tdscf.TDA(mf)
            td.nstates, td.conv_tol, td.max_cycle = cfg["roots"], cfg["td_tol"], cfg["td_max_cycle"]
            td.max_memory, td.frozen = cfg["max_memory"], None
            if cfg["ks"] == "rks":
                td.singlet = True
            active("response_solve", repeat)
            log.timed("response_solve", td.kernel, repeat)
            strengths = log.timed("oscillator_strength", lambda: td.oscillator_strength(gauge="length"), repeat)
            elapsed = time.perf_counter() - start
            energies = np.asarray(td.e)
            converged = np.asarray(td.converged, dtype=bool)
            passed = (energies.size == cfg["roots"] and converged.size == cfg["roots"]
                      and bool(converged.all()) and finite(energies, strengths)
                      and bool(np.all(np.abs(energies.imag) <= 1e-12))
                      and bool(np.all(np.abs(np.asarray(strengths).imag) <= 1e-12)))
            log.add("end_to_end", elapsed, repeat, public_converged=bool(converged.all()),
                response_converged=converged, finite_outputs=finite(energies, strengths), scientific_pass=passed,
                excitations_ha=energies.real, excitation_imaginary_ha=energies.imag,
                oscillator_strength=np.asarray(strengths).real,
                oscillator_strength_imaginary=np.asarray(strengths).imag, actual_roots=energies.size,
                solver_class=f"{type(td).__module__}.{type(td).__name__}",
                cache_state="cold process" if repeat == 0 else "new objects; fixed geometry",
                timing_scope="independent geometry -> SCF -> response -> oscillator strengths",
                notes="Public convergence only; not an independently recomputed original-operator residual certificate")
            metadata = dict(basis=mol._basis, unit=cfg["unit"], atom_coordinates_bohr=mol.atom_coords(),
                cart=mol.cart, ks=cfg["ks"], actual_xc=mf.xc, libxc_rsh_coeff=dft.libxc.rsh_coeff(mf.xc),
                grid_level=mf.grids.level, grid_coordinate_sha256=array_hash(mf.grids.coords),
                grid_weight_sha256=array_hash(mf.grids.weights), grid_dtype=str(mf.grids.weights.dtype),
                grid_settings={key: str(getattr(mf.grids, key, None)) for key in
                    ("prune", "radi_method", "becke_scheme", "radii_adjust", "atom_grid", "alignment")},
                solver_source=inspect.getfile(type(td)), scf_tolerances=dict(energy=mf.conv_tol, gradient=mf.conv_tol_grad),
                response_tolerance=td.conv_tol, nmo=nmo, transition_dimension=dim, **log.context)
            record(f"actual-{repeat}.json", metadata)
            if repeat == 0:
                record("actual-basis.json", mol._basis)
                np.savez(out / "grid.npz", coords=mf.grids.coords, weights=mf.grids.weights)
            if repeat == 3:
                np.savez(out / "states.npz", mo_coeff=mf.mo_coeff, mo_energy=mf.mo_energy,
                    mo_occ=mf.mo_occ, energies=energies, oscillator_strength=strengths)
            if not passed:
                raise ArithmeticError("PySCF response failed requested-root/convergence/finite checks")
            mol.stdout.close()
            del td, mf, mol, occ, energies, strengths, converged, history, callback, metadata
            gc.collect()
        record("clean_phase_done.json", dict(ended_monotonic=time.monotonic(),
            ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            metric="Linux cumulative process high-water mark"))
        record("status.json", dict(status="ok", scientific_pass=True, completed_calls=4,
            original_operator_certificate=False, total_energy_comparable=not original_wb97xd))
        return 0
    except Exception as exc:
        status = "unsupported" if isinstance(exc, NotImplementedError) else "error"
        log.add("failure", None, repeat, status=status, scientific_pass=False,
            exception_type=type(exc).__name__, message=str(exc))
        record("status.json", dict(status=status, scientific_pass=False,
            exception_type=type(exc).__name__, message=str(exc)))
        traceback.print_exc()
        return 2 if status == "unsupported" else 1


if __name__ == "__main__":
    sys.exit(main())
