"""Independent CPU1 references; differing root/algorithm contracts get no ratio."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("pyscf", "molgw"))
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--intel-runtime", type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    case = json.loads(args.case.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    rows = []
    def save(row):
        row.update(engine=args.engine, input_sha256=hashlib.sha256(args.case.read_bytes()).hexdigest(),
                   affinity=sorted(os.sched_getaffinity(0)), time_ratio=None,
                   notes="Independent reference; single-root/upstream diagnostics differ from GradSCF multi-candidate certification")
        rows.append(row)
        with (args.output/"measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
    if args.engine == "pyscf":
        import pyscf
        from pyscf import gto, scf
        from pyscf.gw.gw_cd import GWCD
        pyscf.lib.num_threads(1)
        for repeat in range(args.repeats+1):
            tick = time.perf_counter()
            mol = gto.M(atom=case["atom"], basis=case["basis"], cart=True, verbose=0)
            mf = scf.RHF(mol).density_fit(case["auxbasis"])
            mf.conv_tol = 1e-12
            mf.kernel()
            scf_seconds = time.perf_counter()-tick
            post = time.perf_counter()
            model = GWCD(mf)
            model.with_df = mf.with_df
            model.eta = case["eta_hartree"]
            model.kernel(nw=case["nw"], orbs=case["orbs"], vhf_df=True)
            save(dict(repeat=repeat, wall_seconds=time.perf_counter()-tick,
                scf_seconds=scf_seconds, gw_seconds=time.perf_counter()-post,
                version=pyscf.__version__, nao=mol.nao_nr(),
                bare_exchange="RI; explicit PySCF vhf_df=True",
                hf_energy_hartree=float(mf.e_tot),
                qp_energy_hartree=np.asarray(model.mo_energy)[case["orbs"]].tolist(),
                scf_converged=bool(mf.converged), upstream_qp_converged=bool(model.converged),
                qp_residual_hartree=None, residual_evidence="Not exposed by this installed public GWCD result",
                status="ok" if mf.converged and model.converged else "nonconverged"))
    else:
        if args.executable is None:
            parser.error("--executable is required for MOLGW")
        executable = args.executable.resolve()
        # Only common basis DATA are exported. Each reference still generates
        # its own integrals, orbitals, SCF and QP results independently.
        from gradscf.integrals.basis.data import load_basis_from_snapshot
        basis = args.output/"basis"
        basis.mkdir()
        elements = sorted({row.strip().split()[0] for row in case["atom"].split(";")})
        basis_hashes = {}
        for label, source in (("common-orbital", case["basis"]), ("common-auxiliary", case["auxbasis"])):
            for symbol in elements:
                lines, count = [], 0
                for shell in load_basis_from_snapshot(source, symbol):
                    angular, primitives = int(shell[0]), shell[1:]
                    for column in range(1, len(primitives[0])):
                        count += 1
                        lines.append(f"{len(primitives)} {angular}")
                        lines.extend(f"{row[0]:.16e} {row[column]:.16e}" for row in primitives)
                path = basis/f"{symbol}_{label}"
                path.write_text(str(count)+"\n"+"\n".join(lines)+"\n")
                basis_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output/"basis-data.json").write_text(json.dumps(dict(
            source="GradSCF bundled common basis data only; no calculated electronic inputs",
            original_orbital_basis=case["basis"], original_auxiliary_basis=case["auxbasis"],
            sha256=basis_hashes), indent=2)+"\n")
        environment = {k: v for k, v in os.environ.items() if not k.startswith("MOLGW_COMPARE_")}
        environment["MOLGW_BASIS_PATH"] = str(basis.resolve())+"/"
        environment["OMP_STACKSIZE"] = "256M"
        if args.intel_runtime is not None:
            environment["LD_LIBRARY_PATH"] = str(args.intel_runtime)+":"+environment.get("LD_LIBRARY_PATH", "")
        geometry = [row.strip() for row in case["atom"].split(";")]
        text = ("&molgw\n scf='HF'\n postscf='G0W0'\n gaussian_type='CART'\n"
                " basis='common-orbital'\n auxil_basis='common-auxiliary'\n"
                " tolscf=1.0e-12\n nscf=100\n frozencore='no'\n ncoreg=0\n ncorew=0\n"
                f" nvirtualg={case['nmo_expected']+1}\n nvirtualw={case['nmo_expected']+1}\n"
                f" selfenergy_state_min={min(case['orbs'])+1}\n selfenergy_state_max={max(case['orbs'])+1}\n"
                f" eta={case['eta_hartree']}\n nomega_sigma=40001\n step_sigma=0.00005\n"
                f" length_unit='angstrom'\n natom={len(geometry)}\n/\n"+"\n".join(geometry)+"\n")
        for repeat in range(args.repeats+1):
            work = args.output/f"repeat-{repeat:02d}"
            work.mkdir()
            (work/"input.in").write_text(text)
            tick = time.perf_counter()
            with (work/"stdout.log").open("w") as log:
                result = subprocess.run([str(executable), "input.in"], cwd=work,
                    env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=6000)
            elapsed = time.perf_counter()-tick
            output = (work/"stdout.log").read_text()
            row = dict(repeat=repeat, wall_seconds=elapsed, returncode=result.returncode,
                executable=str(executable), executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                revision="b831818d7a845c36f295d036dc8ceef59daf9991",
                input_file=str(work/"input.in"), normal_end="This is the end" in output,
                status="error", qp_residual_hartree=None)
            path = work/"ENERGY_QP"
            if result.returncode == 0 and path.is_file():
                values = np.atleast_2d(np.loadtxt(path, skiprows=2))
                states = {int(v[0])-1: float(v[1]) for v in values}
                row["qp_energy_hartree"] = [states[p] for p in case["orbs"]]
                row["status"] = "ok" if row["normal_end"] and np.isfinite(row["qp_energy_hartree"]).all() else "nonconverged"
            save(row)
            if row["status"] == "error":
                break
    (args.output/"memory.json").write_text(json.dumps(dict(
        parent_peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        maximum_child_peak_rss_bytes=None if args.engine == "pyscf" else 1024*resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        memory_method="Linux ru_maxrss; child metric is maximum over independent upstream subprocesses"), indent=2)+"\n")


if __name__ == "__main__":
    main()
