"""Isolated TD response measurements. No production kernels are modified.

End-to-end runs create independent SCF objects. Operator runs are explicitly
shared-input diagnostics and never enter an end-to-end speed ratio.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np


def ready(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return value
    seen.add(id(value))
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, dict):
        for item in value.values():
            ready(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            ready(item, seen)
    elif is_dataclass(value):
        for field in fields(value):
            ready(getattr(value, field.name), seen)
    elif callable(value) and getattr(value, "__closure__", None):
        for cell in value.__closure__:
            try:
                contents=cell.cell_contents
            except ValueError:  # Static branch left an unused closure cell unbound.
                continue
            ready(contents, seen)
    return value


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "shape"):
        return plain(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def closed_arrays(function):
    """Logical closed-over arrays, without copying device data or allocating."""
    arrays,seen={},set()
    def visit(value,path):
        if id(value) in seen:return
        seen.add(id(value))
        if hasattr(value,'shape') and hasattr(value,'dtype'):
            arrays[path]=dict(shape=list(value.shape),bytes=int(value.size*value.dtype.itemsize))
        elif is_dataclass(value):
            for field in fields(value):visit(getattr(value,field.name),path+'.'+field.name)
        elif getattr(value,'__closure__',None):
            for name,cell in zip(value.__code__.co_freevars,value.__closure__):
                try:contents=cell.cell_contents
                except ValueError:continue
                visit(contents,path+'.'+name)
    visit(function,'vind')
    return arrays


class Recorder:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.context = dict(nao=None, nocc=None, nvir=None, naux=None, ngrid=None,
                            nroots=config["roots"], batch_size=None)
        self.records = []

    def add(self, stage, seconds, repeat=0, **details):
        row = dict(case_id=self.config["case_id"], variant=self.config["variant"],
                   engine=self.config["engine"], stage=stage, device=self.config["device"],
                   dtype="float64", integral_representation=self.config["representation"],
                   approximation="explicit auxiliary RI" if self.config["representation"] == "df" else "exact",
                   solver_tolerance=self.config["td_tol"], repeat=repeat,
                   wall_seconds=seconds, cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   cpu_memory_metric_kind="Linux cumulative process ru_maxrss; parent also samples RSS",
                   gpu_memory_bytes=None, gpu_memory_metric_kind=None,
                   excitation_energy_difference=None, original_operator_residual=None,
                   gradient_error=None, status="ok", **self.context)
        row.update(details)
        self.records.append(plain(row))
        with (self.output/"measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(plain(row), allow_nan=False)+"\n")

    def timed(self, stage, function, repeat=0, **details):
        start = time.perf_counter()
        value = ready(function())
        self.add(stage, time.perf_counter()-start, repeat, **details)
        return value


def reference(cfg, log, repeat=0):
    if cfg.get('prepare_on_cpu'):
        import jax
        cpu_cfg=dict(cfg,device='cpu',prepare_on_cpu=False)
        with jax.default_device(jax.devices('cpu')[0]):
            mf,ref=reference(cpu_cfg,log,repeat)
        mf.execution_device='gpu'
        known=sum(int(getattr(ref,name).size*getattr(ref,name).dtype.itemsize)
                  for name in ('mo_coeff','mo_occ','mo_energy','ao','ao_deriv1','rdm1','df_factors')
                  if getattr(ref,name,None) is not None)
        ref=log.timed('cpu_reference_to_gpu',mf._ensure_reference,repeat,
                      logical_array_bytes=known,notes='explicit CPU integrals/SCF -> GPU response; bytes are logical estimate, not a bus counter')
        return mf,ref
    options = dict(atom=cfg["atom"], basis=cfg["basis"], unit="Angstrom",
                   charge=cfg.get("charge", 0), spin=cfg.get("spin", 0), cart=True)
    unrestricted = options["spin"] != 0
    if cfg["engine"] == "gradscf":
        from gradscf import dft, gto
        mol = log.timed("molecule", lambda: gto.M(**options), repeat)
        cls = dft.UKS if unrestricted else dft.RKS
        mf = cls(mol, xc=cfg["xc"], grids_level=cfg["grid_level"], execution_device=cfg["device"],
                 conv_tol=1e-10, conv_tol_density=1e-9, conv_tol_grad=1e-8,
                 max_cycle=100, max_memory=4096, init_guess="hcore")
        if cfg["representation"] == "df":
            mf.density_fit(cfg["auxbasis"])
        log.timed("scf", lambda: mf.run().e_tot, repeat,device=cfg['device'])
        ref = mf._ensure_reference()
        ints = mf._scf_inputs if mf._scf_inputs is not None else ref
        naux = None if getattr(ints, "df_factors", None) is None else ints.df_factors.shape[0]
        ngrid = ref.grid.weights.shape[0]
        cycles = mf.cycles
    else:
        from pyscf import dft, gto, lib
        lib.num_threads(cfg["threads"])
        mol = log.timed("molecule", lambda: gto.M(**options, verbose=0), repeat)
        mf = (dft.UKS if unrestricted else dft.RKS)(mol)
        mf.xc, mf.grids.level = cfg["xc"], cfg["grid_level"]
        mf.small_rho_cutoff = 0
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-10, 1e-8, 100
        mf.init_guess, mf.max_memory = "hcore", 4096
        if cfg["representation"] == "df":
            mf = mf.density_fit(auxbasis=cfg["auxbasis"])
        log.timed("scf", mf.kernel, repeat)
        ref = None
        naux = mf.with_df.get_naoaux() if cfg["representation"] == "df" else None
        ngrid, cycles = len(mf.grids.weights), getattr(mf, "cycles", None)
    occ = np.asarray(mf.mo_occ)
    nocc = [int(np.count_nonzero(x)) for x in occ] if unrestricted else int(np.count_nonzero(occ))
    nao = mf.mo_coeff.shape[-1]
    nvir = [nao-n for n in nocc] if unrestricted else nao-nocc
    log.context.update(nao=nao, nocc=nocc, nvir=nvir, naux=naux, ngrid=ngrid)
    def location(value):
        device=getattr(value,'device',None)
        return str(device() if callable(device) else device) if device is not None else 'host/unknown'
    locations = None if cfg['engine'] != 'gradscf' else dict(
        facade_mo=location(mf.mo_coeff),reference_mo=location(ref.mo_coeff),
        grid_weights=location(ref.grid.weights),ao=location(ref.ao),
        df_factors=location(getattr(ref,'df_factors',None)))
    log.add("reference_metadata", None, repeat, scf_energy=float(mf.e_tot),
            scf_iterations=cycles, converged=bool(mf.converged),array_devices=locations)
    if repeat == 0:
        np.savez(log.output/"orbitals.npz", mo_coeff=mf.mo_coeff, mo_energy=mf.mo_energy,
                 mo_occ=mf.mo_occ, energy=mf.e_tot)
        if cfg["engine"] == "pyscf":
            (log.output/"actual-basis.json").write_text(json.dumps(mol._basis, indent=2))
        else:
            log.add("integral_shapes", None, shapes={name: None if getattr(ints,name,None) is None else list(getattr(ints,name).shape)
                    for name in ("eri","eri_pair_matrix","df_factors","ao","ao_deriv1")})
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf, ref


def solver(cfg, mf):
    if cfg["engine"] == "gradscf":
        from gradscf import tdscf
        cls = tdscf.TDDFT if cfg["full"] else tdscf.TDA
        return cls(mf, nstates=cfg["roots"], eigensolver="davidson", davidson_tol=cfg["td_tol"],
                   davidson_max_iter=100,
                   response_kernel_options=dict(two_electron_mode="df" if cfg["representation"] == "df" else "auto",
                                                direct_workers=cfg["threads"]))
    td = mf.TDDFT() if cfg["full"] else mf.TDA()
    td.nstates, td.conv_tol, td.max_cycle = cfg["roots"], cfg["td_tol"], 100
    return td


def end_to_end(cfg, log):
    for repeat in range(4):
        profiler = cProfile.Profile() if cfg.get("profile") else None
        if profiler is not None:
            profiler.enable()
        tick = time.perf_counter()
        mf, ref = reference(cfg, log, repeat)
        td = solver(cfg, mf)
        log.timed("response_solve", td.kernel, repeat)
        oscillator = log.timed("oscillator_strength", td.oscillator_strength, repeat)
        elapsed = time.perf_counter()-tick
        status = "ok" if bool(np.all(td.converged)) else "nonconverged"
        log.add("end_to_end", elapsed, repeat, status=status, excitations_ha=np.asarray(td.e),
                oscillator_strength=oscillator, cache_state="cold new process" if repeat == 0 else "new objects same geometry",
                timing_scope="geometry -> SCF -> response -> oscillator strengths")
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(log.output/f"profile-{repeat}.pstats")


def action(cfg, log):
    import jax
    import jax.numpy as jnp
    mf, ref = reference(cfg, log)
    td = solver(cfg, mf)
    if cfg["engine"] == "gradscf":
        if cfg.get("spin", 0):
            from gradscf.tddft.unrestricted import build_unrestricted_tda_operator
            from gradscf.tddft._unrestricted_semilocal_response import UnrestrictedSemilocalResponseFunctional
            build = lambda: build_unrestricted_tda_operator(ref, UnrestrictedSemilocalResponseFunctional(cfg["xc"]))
            built = log.timed("operator_preparation", build)
            vind, diagonal = built[:2]
            size = int(diagonal.size)
        else:
            from gradscf.tddft.response import build_restricted_tda_operator, build_restricted_tdhf_operator
            from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
            kw = dict(response_kernel_options=dict(two_electron_mode="df" if cfg["representation"] == "df" else "auto",
                                                    direct_workers=cfg["threads"]))
            functional = SemilocalResponseFunctional(cfg["xc"])
            if cfg["full"]:
                vind = log.timed("operator_preparation", lambda: build_restricted_tdhf_operator(ref, functional, **kw))
                size = 2*log.context["nocc"]*log.context["nvir"]
            else:
                vind, diagonal, gaps = log.timed("operator_preparation", lambda: build_restricted_tda_operator(ref, functional, **kw))
                size = gaps.size
        plan = getattr(vind, "memory_plan", None)
        if plan is not None:
            log.context["batch_size"] = plan.batch_size
            log.add("memory_plan", None, plan=dict(plan._asdict()))
        for batch in (1, 5):
            x = jax.device_put(np.random.default_rng(7).normal(size=(batch, size)), jax.devices(cfg["device"])[0])
            compiled_function = jax.jit(vind)
            start = time.perf_counter()
            lowered = compiled_function.lower(x)
            log.add("action_lowering", time.perf_counter()-start, batch_size=batch)
            start = time.perf_counter()
            compiled = lowered.compile()
            log.add("action_compile", time.perf_counter()-start, batch_size=batch,
                    compiler_memory=compiled.memory_analysis()._asdict() if hasattr(compiled.memory_analysis(), "_asdict") else str(compiled.memory_analysis()))
            (log.output/f"hlo-batch{batch}.txt").write_text(compiled.as_text())
            for repeat in range(11):
                out = log.timed("action_first" if repeat == 0 else "action_warm", lambda: compiled(x), repeat,
                                batch_size=batch, timing_scope="prepared operator only; inputs from independent GradSCF SCF")
            np.savez(log.output/f"action-batch{batch}.npz", x=x, action=out)
        ready(vind)
        log.add("operator_resident_arrays",None,closed_arrays=closed_arrays(vind),
                notes="logical closure-array sizes, identity-deduplicated; aliases/compiler DCE can reduce physical storage")
    else:
        vind, diagonal = log.timed("operator_preparation", td.gen_vind)
        for batch in (1, 5):
            x = np.random.default_rng(7).normal(size=(batch, diagonal.size))
            for repeat in range(11):
                out = log.timed("action_first" if repeat == 0 else "action_warm", lambda: vind(x), repeat, batch_size=batch)
            np.savez(log.output/f"action-batch{batch}.npz", x=x, action=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    if cfg['engine']=='gradscf' or cfg['mode']=='action':
        import jax
        jax.config.update("jax_enable_x64", True)
    else:
        jax=None
    log = Recorder(cfg, args.output)
    try:
        if cfg.get("experiment"):
            from experiments import install
            install(cfg)
        fn = end_to_end if cfg["mode"] == "end_to_end" else action
        fn(cfg, log)
        runtime=dict(executable=sys.executable,affinity=sorted(os.sched_getaffinity(0)),
                     slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        if cfg['engine']=='gradscf':
            from gradscf.integrals._native import register_integrals
            lib = Path(register_integrals()._name)
            import gradscf
            runtime.update(gradscf=gradscf.__file__,native=str(lib),
                native_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
                devices=[str(x) for x in jax.devices()],device_memory=[d.memory_stats() for d in jax.devices()])
        else:
            import pyscf
            runtime.update(pyscf=pyscf.__file__,pyscf_version=pyscf.__version__,
                           jax_imported='jax' in sys.modules,gradscf_imported='gradscf' in sys.modules)
        (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2))
    except Exception:
        log.add("failure", None, status="error", notes=traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
