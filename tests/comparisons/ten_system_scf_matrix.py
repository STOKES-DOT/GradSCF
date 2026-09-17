"""Remote-ready HF/DFT matrix with independent PySCF integrals on identical grids."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))

SYSTEMS = {
    "H2": dict(atom="H 0 0 0; H 0 0 .74", charge=0, spin=0),
    "HF": dict(atom="H 0 0 0; F 0 0 .917", charge=0, spin=0),
    "H2O": dict(atom="O 0 0 0; H 0 .757 .587; H 0 -.757 .587", charge=0, spin=0),
    "NH3": dict(atom="N 0 0 .1173; H 0 .9377 -.2737; H .8121 -.46885 -.2737; H -.8121 -.46885 -.2737", charge=0, spin=0),
    "CH4": dict(atom="C 0 0 0; H .629118 .629118 .629118; H -.629118 -.629118 .629118; H -.629118 .629118 -.629118; H .629118 -.629118 -.629118", charge=0, spin=0),
    "CO": dict(atom="C 0 0 0; O 0 0 1.128", charge=0, spin=0),
    "LiH_plus": dict(atom="Li 0 0 0; H 0 0 1.6", charge=1, spin=1),
    "OH": dict(atom="O 0 0 0; H 0 0 .97", charge=0, spin=1),
    "NO": dict(atom="N 0 0 0; O 0 0 1.15", charge=0, spin=1),
    "O2": dict(atom="O 0 0 0; O 0 0 1.21", charge=0, spin=2),
}
XC_REFERENCE = {
    "lda": "LDA_X + LDA_C_PW",
    "svwn": "LDA_X + LDA_C_VWN",
    "pbe": "GGA_X_PBE + GGA_C_PBE",
    "pbe0": "0.25*HF + 0.75*GGA_X_PBE + GGA_C_PBE",
    "b3lyp": "0.20*HF + 0.08*LDA_X + 0.72*GGA_X_B88 + 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP",
}


def json_safe(value):
    if hasattr(value, 'shape') and hasattr(value, 'tolist'):
        return json_safe(value.tolist())
    if isinstance(value, dict): return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def validate_resume(previous, current):
    keys=("system", "basis", "grid_level", "max_cycle", "source_hashes", "benchmark_sha256")
    changed=[key for key in keys if previous.get(key)!=current.get(key)]
    if changed:
        raise ValueError(f"Refusing to reuse results after source/settings changes: {changed}; use a new output directory")


def cases_for(system):
    for method in ("RHF", "UHF", "ROHF", "GHF"):
        yield method, "hf", not (method == "RHF" and system["spin"])
    for xc in XC_REFERENCE:
        for method in ("RKS", "UKS", "ROKS", "GKS"):
            yield method, xc, not (method == "RKS" and system["spin"])
    for xc in ("lda", "svwn"):
        yield "GKS_ncol", xc, True


def prepare(system, basis, grid_level):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.linalg import eigh
    from gradscf import integrals, scf
    from gradscf.data.molecule import parse_molecule_spec
    from pyscf import gto, dft
    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    spec = parse_molecule_spec(system["atom"], charge=system["charge"], spin=system["spin"])
    topology, parameters = integrals.prepare_basis(spec, basis)
    plan = integrals.make_plan(topology, backend="native")
    s = plan.evaluate("overlap", parameters)
    h = plan.evaluate("kinetic", parameters)+plan.evaluate("nuclear", parameters)
    eri = plan.evaluate("eri", parameters)
    enuc = scf.nuclear_repulsion_energy(parameters.nuclear_coords, jnp.asarray(topology.nuclear_charges))
    ao_basis = integrals.basis_from_molecule_spec(spec, basis=basis, precompute_eri_groups=False)
    coords, weights = integrals.build_molecular_grid_from_spec(spec, level=grid_level)
    ao, deriv = integrals.evaluate_cartesian_ao_with_derivatives(ao_basis, coords, deriv=1)
    mol = gto.M(atom=system["atom"], basis=basis, charge=system["charge"], spin=system["spin"], cart=True, unit="Angstrom", verbose=0)
    sr, hr, erir = mol.intor("int1e_ovlp"), mol.intor("int1e_kin")+mol.intor("int1e_nuc"), mol.intor("int2e", aosym="s1")
    aor = dft.numint.eval_ao(mol, np.asarray(coords), deriv=1)
    errors = dict(overlap=float(np.max(np.abs(s-sr))), hcore=float(np.max(np.abs(h-hr))),
                  eri=float(np.max(np.abs(eri-erir))), ao_deriv1=float(np.max(np.abs(deriv-aor))),
                  nuclear_repulsion=abs(float(enuc)-mol.energy_nuc()))
    for key in ("overlap", "hcore", "eri", "nuclear_repulsion"):
        if errors[key] > 3e-10: raise AssertionError((key, errors[key]))
    if errors["ao_deriv1"] > 2e-8: raise AssertionError(errors)
    eps, c = eigh(np.asarray(h), np.asarray(s))
    na, nb = mol.nelec
    da, db = c[:, :na]@c[:, :na].T, c[:, :nb]@c[:, :nb].T
    zero = np.zeros_like(da)
    dspin = np.block([[da,zero],[zero,db]]).astype(complex)
    return dict(s=s,h=h,eri=eri,enuc=enuc,ao=ao,deriv=deriv,coords=np.asarray(coords),weights=np.asarray(weights),
                mol=mol,eri_reference=erir,c=c,da=da,db=db,dspin=dspin,na=na,nb=nb,
                input_errors=errors, preparation_seconds=time.perf_counter()-started)


def run_gradscf(method, xc, p, max_cycle, *, level_shift=0.):
    import jax.numpy as jnp
    import numpy as np
    from gradscf import scf
    common = dict(overlap=p["s"], hcore=p["h"], eri=p["eri"], nuclear_repulsion=p["enuc"])
    controls = dict(max_cycle=max_cycle, conv_tol=1e-11, conv_tol_density=1e-9, level_shift=level_shift)
    spin = dict(nalpha=p["na"], nbeta=p["nb"], init_density_alpha=p["da"], init_density_beta=p["db"])
    grid = dict(ao=p["ao"], ao_deriv1=p["deriv"], grid_weights=jnp.asarray(p["weights"]))
    if method == "RHF":
        result = scf.run_rhf_from_integrals(**common, nelectron=p["na"]+p["nb"],
            config=scf.RHFConfig(**controls))
    elif method == "UHF":
        result = scf.run_uhf_from_integrals(**common, **spin, config=scf.UHFConfig(**controls))
    elif method == "ROHF":
        result = scf.run_rohf_from_integrals(**common, **spin, init_mo_coeff=p["c"], config=scf.ROHFConfig(**controls, conv_tol_grad=1e-7))
    elif method == "GHF":
        result = scf.run_ghf_from_integrals(**common, nelectron=p["na"]+p["nb"], init_density=p["dspin"], config=scf.GHFConfig(**controls, conv_tol_grad=1e-7))
    elif method == "RKS":
        result = scf.run_rks_from_integrals(**common, **grid, nelectron=p["na"]+p["nb"], init_density=p["da"]+p["db"], config=scf.RKSConfig(xc_spec=xc, **controls))
    elif method == "UKS":
        result = scf.run_uks_from_integrals(**common, **grid, **spin, config=scf.UKSConfig(xc_spec=xc, **controls))
    elif method == "ROKS":
        result = scf.run_roks_from_integrals(**common, **grid, **spin, init_mo_coeff=p["c"], config=scf.ROKSConfig(xc_spec=xc, **controls, conv_tol_grad=1e-7))
    else:
        result = scf.run_gks_from_integrals(**common, **grid, nelectron=p["na"]+p["nb"], init_density=p["dspin"], config=scf.GKSConfig(xc_spec=xc, collinear="ncol" if method.endswith("ncol") else "col", **controls, conv_tol_grad=1e-7))
    if method in {"UHF", "UKS", "ROHF", "ROKS"}:
        dm = np.stack([result.density_matrix_alpha, result.density_matrix_beta])
    else:
        dm = np.asarray(result.density_matrix)
    info=dict(energy=float(result.total_energy), converged=bool(result.converged), cycles=int(result.cycles))
    if hasattr(result,"orbital_gradient_norm"):
        info["orbital_gradient_norm"]=float(result.orbital_gradient_norm)
    if method not in {"ROHF","ROKS"}:
        s=np.asarray(p["s"])
        if dm.ndim==3:
            f=np.stack([result.fock_matrix_alpha,result.fock_matrix_beta])
        else:
            f=np.asarray(result.fock_matrix)
            if dm.shape[0]==2*s.shape[0]:s=np.kron(np.eye(2),s)
        info["raw_commutator_max"]=float(np.max(np.abs(f@dm@s-s@dm@f)))
    return info, dm


def run_pyscf(method, xc, p, max_cycle, grid_level, *, level_shift=0., final_cycle=True):
    import numpy as np
    from pyscf import scf, dft
    cls = getattr(scf if xc == "hf" else dft, method.replace("_ncol", ""))
    mf = cls(p["mol"])
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-11, 1e-7, max_cycle
    mf.level_shift=level_shift
    mf.conv_check=final_cycle
    mf.chkfile = None
    mf.direct_scf = False
    mf._eri = p["eri_reference"]
    if xc != "hf":
        mf.xc = XC_REFERENCE[xc]
        mf.grids.level = grid_level
        mf.grids.coords = p["coords"].copy()
        mf.grids.weights = p["weights"].copy()
        mf.small_rho_cutoff = 0.
        if method.startswith("GKS"): mf.collinear = "ncol" if method.endswith("ncol") else "col"
    if method in {"GHF", "GKS", "GKS_ncol"}: dm0 = p["dspin"]
    elif method in {"UHF", "UKS", "ROHF", "ROKS"}: dm0 = np.stack([p["da"], p["db"]])
    else: dm0 = p["da"]+p["db"]
    mf.kernel(dm0=dm0)
    return dict(energy=float(mf.e_tot), converged=bool(mf.converged), cycles=int(mf.cycles)), np.asarray(mf.make_rdm1())


def run_system(args, name):
    import jax
    import numpy as np
    import pyscf
    import importlib.metadata
    output = args.output/name
    output.mkdir(parents=True, exist_ok=True)
    system = SYSTEMS[name]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT/"src/gradscf").rglob("*.py") if "build" not in p.parts and "__pycache__" not in p.parts}
    resume_identity=dict(system=system,basis=args.basis,grid_level=args.grid_level,max_cycle=args.max_cycle,
                         source_hashes=source_hashes,benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.resume and (output/"metadata.json").exists():
        validate_resume(json.loads((output/"metadata.json").read_text()),resume_identity)
    print(f"Preparing {name}: {system}, basis={args.basis}, grid={args.grid_level}", flush=True)
    p = prepare(system, args.basis, args.grid_level)
    metadata = dict(system=system, basis=args.basis, grid_level=args.grid_level, nao=p["s"].shape[0], ngrid=len(p["weights"]),
        input_errors=p["input_errors"], preparation_seconds=p["preparation_seconds"],
        initial_guess="shared hcore orbitals with specified spin occupations", initial_density_source="GradSCF native S/H + scipy.linalg.eigh",
        xc_reference=XC_REFERENCE, max_cycle=args.max_cycle, conv_tol=1e-11, conv_tol_density=1e-9, conv_tol_grad_reference=1e-7,
        tolerances=dict(hf_energy=1e-8,dft_energy=1e-6,electron_count=1e-6),
        host=platform.node(),platform=platform.platform(),jax=jax.__version__,pyscf=pyscf.__version__,
        jax_xc=importlib.metadata.version("jax-xc"), devices=[str(d) for d in jax.devices()],dtype="float64",
        source_hashes=source_hashes,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (ROOT/"src/gradscf/integrals/_native/csrc").iterdir() if f.is_file()},
        vendor_manifest_sha256=hashlib.sha256((ROOT/"src/gradscf/integrals/_native/vendor/manifest.json").read_bytes()).hexdigest())
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    result_path=output/"results.jsonl"
    completed = {r["case"] for r in (json.loads(line) for line in result_path.read_text().splitlines())} if args.resume and result_path.exists() else set()
    if not args.resume: result_path.write_text("")
    for method, xc, applicable in cases_for(system):
        case=method+"_"+xc
        if args.methods and method not in args.methods: continue
        if case in completed: continue
        row=dict(system=name,method=method,xc=xc,case=case)
        if not applicable:
            row.update(status="not_applicable",reason="Restricted closed-shell method requires spin=0")
        else:
            try:
                started=time.perf_counter();g,dg=run_gradscf(method,xc,p,args.max_cycle);row["gradscf_seconds"]=time.perf_counter()-started
                started=time.perf_counter();r,dr=run_pyscf(method,xc,p,args.max_cycle,args.grid_level);row["pyscf_seconds"]=time.perf_counter()-started
                s=np.asarray(p["s"])
                metric=np.kron(np.eye(2),s) if method in {"GHF","GKS","GKS_ncol"} else s
                dg_total=dg.sum(axis=0) if dg.ndim==3 else dg
                dr_total=dr.sum(axis=0) if dr.ndim==3 else dr
                ng=float(np.trace(dg_total@metric).real);nr=float(np.trace(dr_total@metric).real)
                error=abs(g["energy"]-r["energy"]);tol=1e-8 if xc=="hf" else 1e-6
                finite=bool(np.isfinite(g["energy"]) and np.isfinite(r["energy"]) and np.all(np.isfinite(dg)) and np.all(np.isfinite(dr)))
                status="pass" if finite and g["converged"] and r["converged"] and error<=tol and max(abs(ng-p["mol"].nelectron),abs(nr-p["mol"].nelectron))<=1e-6 else "energy_mismatch"
                if not finite: status="nonfinite"
                elif not g["converged"] or not r["converged"]: status="not_converged"
                row.update(status=status,gradscf=g,pyscf=r,abs_energy_error=error,energy_tolerance=tol,
                    density_max_error=float(np.max(np.abs(dg-dr))),gradscf_electrons=ng,pyscf_electrons=nr)
                if dg.ndim==3:
                    row.update(gradscf_spin_z=float(np.trace((dg[0]-dg[1])@s).real/2),pyscf_spin_z=float(np.trace((dr[0]-dr[1])@s).real/2))
                elif method in {"GHF","GKS","GKS_ncol"}:
                    n=s.shape[0]
                    row.update(gradscf_spin_z=float(np.trace((dg[:n,:n]-dg[n:,n:])@s).real/2),pyscf_spin_z=float(np.trace((dr[:n,:n]-dr[n:,n:])@s).real/2))
                np.savez_compressed(output/(case+".npz"),density_gradscf=dg,density_pyscf=dr,overlap=s)
            except Exception as exc:
                row.update(status="error",error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        with result_path.open("a") as f: f.write(json.dumps(json_safe(row),allow_nan=False)+"\n")
        print(name,case,row["status"],f"dE={row.get('abs_energy_error',float('nan')):.3e}",flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems",nargs="+",choices=SYSTEMS,default=list(SYSTEMS))
    parser.add_argument("--worker",choices=SYSTEMS)
    parser.add_argument("--methods",nargs="+")
    parser.add_argument("--basis",default="3-21g")
    parser.add_argument("--grid-level",type=int,default=1)
    parser.add_argument("--max-cycle",type=int,default=200)
    parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("artifacts/ten-system-scf"))
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.worker:return run_system(args,args.worker)
    started=time.perf_counter()
    def worker(name):
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker",name,"--basis",args.basis,"--grid-level",str(args.grid_level),"--max-cycle",str(args.max_cycle),"--output",str(args.output)]
        if args.resume:cmd.append("--resume")
        if args.methods:cmd += ["--methods",*args.methods]
        with (args.output/(name+".log")).open("a" if args.resume else "w") as log:
            try:return dict(system=name,returncode=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode)
            except subprocess.TimeoutExpired:return dict(system=name,returncode="timeout")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool: workers=list(pool.map(worker,args.systems))
    rows=[]
    for name in args.systems:
        path=args.output/name/"results.jsonl"
        if path.exists(): rows.extend(json.loads(line) for line in path.read_text().splitlines())
    counts={status:sum(r["status"]==status for r in rows) for status in sorted({r["status"] for r in rows})}
    summary=dict(workers=workers,counts=counts,rows=rows,elapsed_seconds=time.perf_counter()-started,
                 requested_systems=args.systems,expected_rows=sum(sum(not args.methods or m in args.methods for m,x,a in cases_for(SYSTEMS[n])) for n in args.systems))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"},indent=2),flush=True)
    return int(any(w["returncode"]!=0 for w in workers) or len(rows)!=summary["expected_rows"])


if __name__=="__main__":raise SystemExit(main())
