"""Saved-reference stationarity/Fock probes, optionally reconverging zero-screen SCF; no R derivative."""
import argparse
import hashlib
import inspect
import json
import os
import resource
from pathlib import Path
from time import perf_counter

parser = argparse.ArgumentParser(description=__doc__)
for name in ("checkpoint", "case", "jsonl", "output"):
    parser.add_argument(name, type=Path)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--fock-only", action="store_true")
parser.add_argument("--rks-only", action="store_true")
parser.add_argument("--solve-zero", action="store_true")
parser.add_argument("--fock-repeats", type=int, default=0)
parser.add_argument("--reconverged", type=Path)
parser.add_argument("--grid-block-size", type=int, default=0)
parser.add_argument("--diagnostic-cycles", type=int, default=0)
parser.add_argument("--preserve-origin", action="store_true")
args = parser.parse_args()
allowed = sorted(os.sched_getaffinity(0))
assert 0 < args.threads <= len(allowed)
os.sched_setaffinity(0, allowed[:args.threads])
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["GRADSCF_NATIVE_THREADS"] = str(args.threads)
case = json.loads(args.case.read_text())
rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
reference = next(row for row in reversed(rows) if row.get("stage") == "reference")
complete = next(row for row in reversed(rows) if row.get("stage") == "end_to_end")
manifest = json.loads(args.jsonl.with_suffix(".manifest.json").read_text())
assert reference["scientific_pass"] is True and complete["scientific_pass"] is True
assert reference["xc"] == complete["xc"] == "pbe0" and complete["method"] == "g0w0"
assert reference["basis"] == complete["basis"] == case["basis"]
assert reference["case_id"] == complete["case_id"] == case["case_id"]
assert (reference["repeat"], reference["grid_level"]) == (complete["repeat"], complete["grid_level"])
assert manifest["atom"] == case["atom"] and manifest["unit"] == case.get("unit", "Angstrom")
assert Path(manifest["arguments"]["checkpoint"]).name == args.checkpoint.name

import jax
import jax.numpy as jnp
import numpy as np
from gradscf import dft, gto, scf
from gradscf_tools.molecular_response import MolecularEnergy

with np.load(args.checkpoint, allow_pickle=False) as archive:
    saved = {name: np.array(archive[name]) for name in archive.files}
reconverged_source = None
if args.reconverged:
    lineage = json.loads(args.reconverged.with_suffix(".json").read_text())
    assert lineage["input_sha256"]["checkpoint"] == hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert lineage["zero_screening_scf"]["converged"] is True
    assert lineage["zero_screening_scf"]["residual_norm"] <= 1e-10
    with np.load(args.reconverged, allow_pickle=False) as archive:
        saved = {name: np.array(archive[name]) for name in archive.files}
    reconverged_source = dict(path=str(args.reconverged),
        sha256=hashlib.sha256(args.reconverged.read_bytes()).hexdigest())
    reference["energy_hartree"] = lineage["zero_screening_scf"]["energy"]
assert saved["mo_coeff"].shape[0] == reference["nao"] == complete["nao"]
mol = gto.M(atom=case["atom"], basis=case["basis"], cart=case["cart"],
            unit=case["unit"], charge=case.get("charge", 0), spin=case.get("spin", 0))
mf = dft.RKS(mol, xc=reference["xc"], grids_level=reference["grid_level"],
    conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
    max_cycle=args.diagnostic_cycles or 200,
    grid_block_size=args.grid_block_size).direct_scf()
mf.mo_coeff, mf.mo_occ, mf.mo_energy = (jnp.asarray(saved[name]) for name in ("mo_coeff", "mo_occ", "mo_energy"))
mf.converged = True  # Verified saved provenance; context for input assembly only.
mf.direct_scf_workers = args.threads
model = MolecularEnergy(mf, jk_backend="direct", max_memory=60000.)
started = perf_counter()
inputs = jax.block_until_ready(model.inputs(model.coordinates))
assembly_seconds = perf_counter() - started
zero_scf = None
if args.solve_zero:
    from gradscf.scf.orbital_optimization import _orthonormalize
    from gradscf.scf.rks import run_rks_from_integrals_traceable, _mo_residual_norm
    if args.diagnostic_cycles:
        # Separate instrumentation run: callback timings are not clean solver
        # timings. The original production step supplies every updated array.
        from gradscf.scf import rks
        original, counter = rks._advance_scf_iteration_with_fock_builder, [0]
        def event(stage, converged, change, norm):
            counter[0] += stage == "iteration"
            row = dict(stage=stage, iteration=counter[0], converged=bool(converged),
                energy_change=float(change) if np.isfinite(change) else None,
                orbital_norm=float(norm) if np.isfinite(norm) else None)
            with args.output.with_suffix(".scf-events.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
        def watched(**options):
            builder = options["energy_and_fock_builder"]
            def observed(*values):
                result = builder(*values)
                if values[4] is None:
                    norm = _mo_residual_norm(result[2], values[1], values[2])
                    jax.debug.callback(lambda e, n: event("full_refresh", False, e, n),
                        result[0]-options["energy"], norm, ordered=True)
                return result
            result = original(**{**options, "energy_and_fock_builder": observed})
            jax.debug.callback(lambda c, e, n: event("iteration", c, e, n), result[0],
                result[4]-options["energy"],
                _mo_residual_norm(result[6], result[2], options["mo_occ_fixed"]), ordered=True)
            return result
        rks._advance_scf_iteration_with_fock_builder = watched
    started = perf_counter()
    zero = jax.block_until_ready(run_rks_from_integrals_traceable(
        **inputs, nelectron=int(np.sum(saved["mo_occ"])),
        init_mo_coeff=_orthonormalize(mf.mo_coeff[None], inputs["overlap"])[0],
        config=model.scf_config))
    zero_scf = dict(converged=bool(zero.converged), cycles=int(zero.cycles),
        energy=float(zero.total_energy), seconds=perf_counter()-started,
        residual_norm=float(_mo_residual_norm(zero.fock_matrix, zero.mo_coeff, zero.mo_occ)),
        tolerance_energy=model.scf_config.conv_tol, tolerance_density=model.scf_config.conv_tol_density,
        tolerance_gradient=model.scf_config.conv_tol_grad,
        direct_cutoff=model.scf_config.direct_scf_tol)
    print(json.dumps(dict(zero_screening_scf=zero_scf)), flush=True)
    np.savez(args.output.with_suffix(".npz"), mo_coeff=np.asarray(zero.mo_coeff),
        mo_occ=np.asarray(zero.mo_occ), mo_energy=np.asarray(zero.mo_energy),
        density=np.asarray(zero.density_matrix), fock=np.asarray(zero.fock_matrix),
        overlap=np.asarray(inputs["overlap"]))
    if not zero_scf["converged"]:
        args.output.write_text(json.dumps(dict(zero_screening_scf=zero_scf), indent=2)+"\n")
        raise SystemExit(2)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = zero.mo_coeff, zero.mo_occ, zero.mo_energy
if args.fock_only or args.rks_only or args.fock_repeats:
    from gradscf.scf import orbital_optimization as orbital
    from gradscf.scf.uks import UKSConfig
    topology = tuple(tuple(float(x) for x in row) for row in model.occupations)
    _, _, density_from, evaluate, _ = orbital._problem_functions(
        "roks", mf.xc, "col", topology, uks_config=UKSConfig(
            xc_spec=mf.xc, jk_backend="direct", direct_scf_workers=args.threads))
    physical = dict(metric=inputs["overlap"], **{key: value for key, value in inputs.items()
                                              if key != "overlap"})
    if args.fock_repeats:
        from gradscf.integrals._native import register_integrals
        def event(**row):
            with args.output.with_suffix(".events.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
        function = jax.jit(evaluate)
        density = density_from(mf.mo_coeff[None])
        started = perf_counter()
        lowered = function.lower(density, physical)
        lower_seconds = perf_counter()-started
        event(stage="lower", seconds=lower_seconds)
        started = perf_counter()
        executable = lowered.compile()
        compile_seconds = perf_counter()-started
        event(stage="compile", seconds=compile_seconds,
              xla_temp_bytes=executable.memory_analysis().temp_size_in_bytes)
        samples = []
        for repeat in range(args.fock_repeats+1):
            started = perf_counter()
            value, fock = jax.block_until_ready(executable(density, physical))
            samples.append(dict(repeat=repeat, seconds=perf_counter()-started, energy=float(value)))
            event(stage="fock", **samples[-1])
        library = Path(register_integrals()._name).resolve()
        source = Path(inspect.getsourcefile(evaluate)).resolve()
        np.savez(args.output.with_suffix(".fock.npz"), fock=np.asarray(fock), density=np.asarray(density))
        payload = dict(scope="Shared prepared-input Fock only; no SCF or derivative", samples=samples,
            lower_seconds=lower_seconds, compile_seconds=compile_seconds,
            xla_temp_bytes=executable.memory_analysis().temp_size_in_bytes,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            assembly_seconds=assembly_seconds, affinity=sorted(os.sched_getaffinity(0)),
            source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            native_path=str(library), native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
        args.output.write_text(json.dumps(payload, indent=2)+"\n")
        print(json.dumps(payload, indent=2), flush=True)
        raise SystemExit(0)
    records = []
    coefficients = mf.mo_coeff[None]
    for normalizations in range(0 if args.rks_only else 2):
        started = perf_counter()
        density = density_from(coefficients)
        value, fock = jax.block_until_ready(evaluate(density, physical))
        block = coefficients[0]
        occupied, virtual = np.flatnonzero(saved["mo_occ"]), np.flatnonzero(saved["mo_occ"] == 0)
        residual = 2*block[:, virtual].T @ fock.mean(axis=0) @ block[:, occupied]
        sym = .5*(fock+fock.swapaxes(-1, -2))
        fmo = jnp.einsum("spi,spq,sqj->sij", jnp.broadcast_to(coefficients, sym.shape),
                         sym, jnp.broadcast_to(coefficients, sym.shape), precision="highest")
        occupation = jnp.asarray(model.occupations)
        pairs = np.where(np.triu(np.any(np.asarray(occupation)[:, :, None]
                                       != np.asarray(occupation)[:, None, :], axis=0), 1))
        body = (2*(occupation[:, None, :]-occupation[:, :, None])*fmo).sum(axis=0)[pairs]
        records.append(dict(normalizations=normalizations, energy=float(value),
            residual_norm=float(jnp.linalg.norm(residual)), seconds=perf_counter()-started,
            body_gradient_norm=float(.5*jnp.linalg.norm(body)),
            density_difference=float(jnp.max(jnp.abs(density.sum(axis=0)-saved["density"]))),
            fock_difference=float(jnp.max(jnp.abs(fock.mean(axis=0)-saved["fock"])))))
        coefficients = orbital._orthonormalize(coefficients, inputs["overlap"])
    if args.rks_only:
        from gradscf.scf.rks import RKSConfig, _make_jk_builder, _energy_and_raw_fock_for_density
        cfg = RKSConfig(xc_spec=mf.xc, jk_backend="direct", direct_scf_workers=args.threads)
        builder = _make_jk_builder(None, cfg, direct_basis=inputs["direct_basis"])
        started = perf_counter()
        value, _, fock, _, _ = jax.block_until_ready(_energy_and_raw_fock_for_density(
            density=jnp.asarray(saved["density"]), mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ,
            mo_energy=mf.mo_energy, jk_builder=builder, ao=inputs["ao"],
            ao_deriv1=inputs["ao_deriv1"], weights=inputs["grid_weights"], h=inputs["hcore"],
            enuc=inputs["nuclear_repulsion"], alpha=jnp.asarray(.25), cfg=cfg, xc_kind="GGA"))
        occupied, virtual = np.flatnonzero(saved["mo_occ"]), np.flatnonzero(saved["mo_occ"] == 0)
        residual = 2*mf.mo_coeff[:, virtual].T @ fock @ mf.mo_coeff[:, occupied]
        records.append(dict(path="fresh RKS/PBE0 J/K and XC", energy=float(value),
            residual_norm=float(jnp.linalg.norm(residual)), seconds=perf_counter()-started,
            fock_difference=float(jnp.max(jnp.abs(fock-saved["fock"])))))
    payload = dict(scope="Shared saved-reference Fock reevaluation; no SCF or derivative", rows=records,
        reconverged_source=reconverged_source,
        assembly_seconds=assembly_seconds, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload, indent=2), flush=True)
    raise SystemExit(0)
started = perf_counter()
state = jax.block_until_ready(scf.minimize_roks_from_integrals(**inputs, xc_spec=mf.xc,
    mo_coeff=mf.mo_coeff, mo_occ=model.occupations, max_iterations=0, gradient_tolerance=1e-10,
    orthonormalize_initial=not args.preserve_origin,
    differentiation=scf.SCFDifferentiationConfig(mode="implicit"),
    direct_scf_workers=args.threads))
primal_seconds = perf_counter() - started


def orbital_norm(fock, coefficients):
    fock = np.broadcast_to(np.asarray(fock), (2, coefficients.shape[0], coefficients.shape[0]))
    fmo = coefficients.T @ fock @ coefficients
    occ = np.asarray(model.occupations)
    pairs = np.where(np.triu(np.any(occ[:, :, None] != occ[:, None, :], axis=0), 1))
    gradient = (2 * (occ[:, None, :] - occ[:, :, None]) * fmo).sum(axis=0)
    return float(.5 * np.linalg.norm(gradient[pairs]))


c, s, f = map(np.asarray, (state.mo_coeff, inputs["overlap"], state.fock_matrix))
maximum = lambda value: float(np.max(np.abs(value)))
sources = {name: Path(inspect.getsourcefile(obj)).resolve() for name, obj in
           (("orbital", scf.minimize_roks_from_integrals), ("assembly", MolecularEnergy))}
from gradscf.scf import rks as rks_module
from gradscf.integrals._native import register_integrals
sources["rks"] = Path(rks_module.__file__).resolve()
library = Path(register_integrals()._name).resolve()
payload = dict(scope=__doc__, original_reference_converged=reference["scientific_pass"],
    zero_screening_scf=zero_scf,
    reconverged_source=reconverged_source, instrumented=bool(args.diagnostic_cycles),
    preserve_origin=args.preserve_origin,
    grid_block_size=args.grid_block_size, native_path=str(library),
    native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    restored_context_only=True, xc=mf.xc, basis=case["basis"], nao=c.shape[0], units="Hartree; Gram dimensionless",
    gradient_norm=float(state.gradient_norm), stationary=bool(state.stationary), iterations=int(state.iterations),
    process_status="ok", scientific_pass=bool(state.stationary),
    total_energy=float(state.total_energy), saved_rks_energy=reference["energy_hartree"],
    energy_difference=float(state.total_energy) - reference["energy_hartree"],
    reported_fock_orbital_norm=orbital_norm(f, c), saved_fock_orbital_norm=orbital_norm(saved["fock"], saved["mo_coeff"]),
    fock_difference_maxabs=maximum(f.mean(axis=0) - saved["fock"]),
    saved_gram_error=maximum(saved["mo_coeff"].T @ saved["overlap"] @ saved["mo_coeff"] - np.eye(c.shape[1])),
    current_gram_error=maximum(c.T @ s @ c - np.eye(c.shape[1])), overlap_difference_maxabs=maximum(s - saved["overlap"]),
    assembly_seconds=assembly_seconds, primal_seconds=primal_seconds, affinity=sorted(os.sched_getaffinity(0)),
    source={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in sources.items()},
    input_sha256={name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() for name in ("checkpoint", "case", "jsonl")})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2), flush=True)
