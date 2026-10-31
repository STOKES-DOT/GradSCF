"""Bounded, synchronized G0W0 probes; run each variant in a fresh process."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("residue", "residue-ad", "water"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--naux", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--nw", type=int, default=32)
    parser.add_argument("--method", choices=("g0w0", "evgw", "evgw0"), default="g0w0")
    parser.add_argument("--conventional", choices=("packed", "direct"))
    parser.add_argument("--max-memory", type=float, default=60000.)
    parser.add_argument("--case-input", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--screening-solver", choices=("direct", "gmres"), default="direct")
    args = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    if args.threads > len(allowed):
        raise ValueError("Requested threads exceed the Slurm step CPU affinity")
    os.sched_setaffinity(0, allowed[:args.threads])
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    target_device = jax.devices(args.device)[0] if args.device else jax.devices()[0]
    jax.config.update("jax_default_device", target_device)
    from gradscf.gw.self_energy import sigma_residue_part

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(python=sys.executable, source=gradscf.__file__,
                    source_sha256=hashlib.sha256(Path(sigma_residue_part.__code__.co_filename).read_bytes()).hexdigest(),
                    versions={p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "numpy", "scipy")},
                    devices=list(map(str, jax.devices())), device_kinds={str(d): d.device_kind for d in jax.devices()}, affinity=sorted(os.sched_getaffinity(0)),
                    threads={p: os.getenv(p) for p in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GRADSCF_NATIVE_THREADS")},
                    mode=args.mode, method=args.method, conventional=args.conventional,
                    repeats=args.repeats, nw=args.nw,
                    source_hashes={str(p.relative_to(Path(gradscf.__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(gradscf.__file__).parent/"gw").glob("*.py")},
                    case_input_sha256=None if args.case_input is None else hashlib.sha256(args.case_input.read_bytes()).hexdigest(),
                    compilation_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
                    dtype="float64", native_libraries=[])
    def record(**row):
        with args.output.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    from gradscf.integrals._native import register_integrals
    handle = register_integrals()
    actual_library = Path(handle._name).resolve()
    manifest["loaded_native_library"] = dict(path=str(actual_library),
        sha256=hashlib.sha256(actual_library.read_bytes()).hexdigest())
    topology = []
    for logical in os.sched_getaffinity(0):
        base = Path(f"/sys/devices/system/cpu/cpu{logical}/topology")
        topology.append(dict(logical_cpu=logical, physical_package=(base/"physical_package_id").read_text().strip(),
                             core_id=(base/"core_id").read_text().strip()))
    manifest["cpu_topology"] = topology
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    if args.mode.startswith("residue"):
        rng = np.random.default_rng(31)
        nmo, nocc = 32, 8
        energy = jnp.asarray(np.r_[np.linspace(-1.8, -.3, nocc), np.linspace(.12, 1.8, nmo-nocc)])
        ov = jnp.asarray(rng.normal(size=(args.naux, nocc, nmo-nocc))*.025/np.sqrt(args.naux))
        left = jnp.asarray(rng.normal(size=(args.naux, nmo))*.1/np.sqrt(args.naux))
        right = jnp.asarray(rng.normal(size=(args.naux, nmo))*.1/np.sqrt(args.naux))
        def evaluate(omega, scale):
            return sigma_residue_part(omega, energy, left*scale, right,
                ((energy, ov*scale, 2.),), -.09, .01)
        target = jax.jit(evaluate if args.mode == "residue" else
            lambda omega, scale: jax.value_and_grad(lambda s: evaluate(omega, s).real)(scale))
        tick = time.perf_counter()
        lowered = target.lower(jnp.asarray(.86), jnp.asarray(1.))
        lower_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        executable = lowered.compile()
        compile_seconds = time.perf_counter()-tick
        analysis = executable.memory_analysis()
        for repeat in range(args.repeats+1):
            tick = time.perf_counter()
            value = jax.block_until_ready(executable(jnp.asarray(.86), jnp.asarray(1.)))
            record(case_id="synthetic-residue", stage="first_execution" if repeat == 0 else "warm_execution",
                   repeat=repeat, wall_seconds=time.perf_counter()-tick,
                   naux=args.naux, nmo=nmo, nocc=nocc, nvir=nmo-nocc,
                   value_real=float(value.real) if args.mode == "residue" else float(value[0]),
                   value_imag=float(value.imag) if args.mode == "residue" else None,
                   parameter_gradient=None if args.mode == "residue" else float(value[1]),
                   status="ok" if all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(value)) else "error")
        record(case_id="synthetic-residue", stage="compile", lower_seconds=lower_seconds,
               wall_seconds=compile_seconds, naux=args.naux,
               xla_temp_bytes=getattr(analysis, "temp_size_in_bytes", None),
               xla_argument_bytes=getattr(analysis, "argument_size_in_bytes", None),
               xla_output_bytes=getattr(analysis, "output_size_in_bytes", None), status="ok")
    else:
        from gradscf import gto, scf
        from gradscf.gw.rgw import _run_g0w0_numerics
        case = json.loads(args.case_input.read_text()) if args.case_input else {}
        atom = case.get("atom", "O 0 0 .117790; H 0 .755453 -.471161; H 0 -.755453 -.471161")
        for repeat in range(args.repeats+1):
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.alarm(6000)
            total = time.perf_counter()
            with jax.default_device(jax.devices("cpu")[0]):
                mol = gto.M(atom=atom, basis=case.get("basis", "def2-svp"), cart=True)
                mf = scf.RHF(mol, conv_tol=1e-12)
                if args.conventional == "direct":
                    mf = mf.direct_scf()
                elif args.conventional is None:
                    mf = mf.density_fit(case.get("auxbasis", "weigend"))
                mf.run()
            jax.block_until_ready(mf.scf_result)
            scf_seconds = time.perf_counter()-total
            r = mf.scf_result
            nocc = int(np.count_nonzero(np.asarray(r.mo_occ)))
            requested = np.array(case.get("orbs", [3, 4, 5, 6]), dtype=int)
            tick = time.perf_counter()
            source = ({"eri_pair_matrix": mf._scf_inputs.eri_pair_matrix} if args.conventional == "packed"
                else {"direct_basis": mf._scf_inputs.direct_basis} if args.conventional == "direct"
                else {"df_factors": mf._scf_inputs.df_factors})
            inputs = (r.mo_energy, r.mo_coeff, source,
                      r.fock_matrix, r.hcore_matrix, r.density_matrix)
            transfer_bytes = sum(x.nbytes for x in jax.tree.leaves(inputs) if getattr(x, "device", None) != target_device)
            energy, coefficients, source, fock, hcore, density = jax.block_until_ready(jax.device_put(inputs, target_device))
            transfer_seconds = time.perf_counter()-tick
            controls = {}
            if args.screening_solver == "gmres":
                from gradscf.solvers import LinearSolverConfig
                controls["screening_config"] = LinearSolverConfig(method="gmres", rtol=1e-11, atol=1e-13, restart=40)
            driver = _run_g0w0_numerics
            if args.method != "g0w0":
                from gradscf.gw.evgw import evgw_cd_restricted
                driver = evgw_cd_restricted
                controls.update(max_iter=60, tol=1e-8, damping=.2, update_w=args.method == "evgw")
            else:
                controls.update(qp_tol=1e-10, resolvent_expansion=False)
            if args.conventional:
                controls["max_memory"] = args.max_memory
            tick = time.perf_counter()
            result = driver(mo_energy=energy, mo_coeff=coefficients, nocc=nocc,
                **source, fock_matrix=fock,
                hcore_matrix=hcore, density_matrix=density,
                nw=args.nw, eta=case.get("eta_hartree", .001), orbs=tuple(map(int, requested)), **controls)
            jax.block_until_ready(result)
            signal.alarm(0)
            gw_seconds = time.perf_counter()-tick
            passed = (bool(mf.converged) and bool(np.asarray(result.converged_mask)[requested].all())
                      and bool(np.asarray(result.qp_branch_valid)[requested].all())
                      and bool(np.asarray(result.qp_computed_mask)[requested].all())
                      and bool(np.all(np.isfinite(np.asarray(result.qp_weight)[requested])))
                      and bool(np.all(np.asarray(result.qp_weight)[requested] > 0))
                      and float(np.max(np.abs(np.asarray(result.qp_residual)[requested]))) < (1e-10 if args.method == "g0w0" else 1e-8))
            record(case_id=case.get("case_id", "water-def2-svp-true-ri-direct-cd"), stage="end_to_end", method=args.method, integral_representation=args.conventional or "ri",
                   repeat=repeat, wall_seconds=time.perf_counter()-total,
                   scf_seconds=scf_seconds, gw_seconds=gw_seconds, transfer_seconds=transfer_seconds,
                   transfer_bytes_estimate=transfer_bytes, screening_solver=args.screening_solver, device=str(target_device),
                   nao=int(r.mo_coeff.shape[0]), naux=None if args.conventional else int(mf._scf_inputs.df_factors.shape[0]),
                   nocc=nocc, nvir=int(r.mo_coeff.shape[1])-nocc, nw=args.nw,
                   mf_energy_hartree=np.asarray(r.mo_energy)[requested].tolist(),
                   qp_energy_hartree=np.asarray(result.mo_energy)[requested].tolist(),
                   qp_residual_hartree=np.asarray(result.qp_residual)[requested].tolist(),
                   qp_weight=np.asarray(result.qp_weight)[requested].tolist(),
                   qp_candidate_count=np.asarray(result.qp_candidate_count)[requested].tolist(),
                   qp_branch_valid=np.asarray(result.qp_branch_valid)[requested].tolist(),
                   scientific_pass=passed, status="ok" if passed else "nonconverged")
        for path in Path(gradscf.__file__).parent.glob("integrals/_native/lib/*.so"):
            manifest["native_libraries"].append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    record(stage="process_peak", cpu_peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           memory_metric="Linux ru_maxrss; includes preparation/compilation/execution", status="ok")
    manifest["gpu_memory"] = []
    for device in jax.devices():
        if device.platform == "cpu":
            continue
        statistics = device.memory_stats() or {}
        manifest["gpu_memory"].append(dict(device=str(device), metrics={
            key: int(statistics[key]) if statistics.get(key) is not None else None
            for key in ("bytes_in_use", "peak_bytes_in_use", "pool_bytes", "peak_pool_bytes", "bytes_limit")},
            notes="Allocator counters; pool/reserved is not active allocation. Missing counters are unknown."))
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
