"""Freeze the balanced 60-system SCF matrix before numerical execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METHODS = ("RHF", "UHF", "GHF", "RKS", "UKS", "GKS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output/"configs"
    inputs.mkdir()
    catalog_path = ROOT/"systems.json"
    catalog = json.loads(catalog_path.read_text())
    fixtures = json.loads((ROOT/"fixture_manifest.json").read_text())
    functionals = catalog["protocol"]["functionals"]
    pbe = next(item for item in functionals if item["gradscf"] == "pbe")
    systems = catalog["systems"]
    by_id = {item["id"]:item for item in systems}
    bases = list(fixtures["primary_basis_counts"])
    definitions = []
    # Each basis gets one member of every XC family in each size quartile.
    # The largest member of a quartet is PBE; this ordering is fixed before runs.
    pattern = [functionals[index] for index in (3, 2, 0, 1)]
    for basis in bases:
        bucket = sorted((s for s in systems if s["basis"] == basis), key=lambda s:(s["natom"],s["id"]))
        assert len(bucket) == 12
        for i, system in enumerate(bucket):
            definitions.append((system, basis, pattern[i % 4], "primary", 1))
    for control in fixtures["optional_cross_basis_controls"]:
        for basis in control["basis"]:
            definitions.append((by_id[control["id"]], basis, pbe, "cross_basis", 4))
    for name in ("methane", "benzene", "phenanthrene"):
        for functional in functionals:
            definitions.append((by_id[name], by_id[name]["basis"], functional, "cross_xc", 4))
    calculations = {}
    for system, basis, functional, scope, repeats in definitions:
        for method in METHODS:
            xc = functional["gradscf"] if method.endswith("KS") else "hf"
            for engine in ("gradscf", "pyscf"):
                key = (system["id"],basis,method,xc,engine)
                if key in calculations:
                    row = calculations[key]
                    row["repeats"] = max(row["repeats"], repeats)
                    if scope not in row["scopes"]:
                        row["scopes"].append(scope)
                    continue
                case_id = "-".join((system["id"],basis,method.lower(),xc,engine)).replace("*","s")
                threads = 64 if system["natom"] >= 30 or (basis == "def2-tzvp" and system["natom"] >= 15) else 8
                calculations[key] = dict(system, case_id=case_id, basis=basis, engine=engine,
                    method=method, xc=xc, pyscf_xc=functional["pyscf"] if method.endswith("KS") else "hf",
                    xc_family=functional["family"] if method.endswith("KS") else "HF",
                    scopes=[scope], repeats=repeats, threads=threads, max_memory=180000.,
                    grid_level=3, max_cycle=200, conv_tol=1e-10, density_tol=1e-8, gradient_tol=1e-7,
                    representation="auto", ri_ao_threshold=128, auxbasis="weigend",
                    single_call_limit_seconds=6000 if threads == 64 else 1800,
                    energy_comparison_tolerance_hartree=1e-7)
    ordered = sorted(calculations.values(), key=lambda row:(
        "primary" not in row["scopes"], row["natom"],row["id"],row["basis"],row["method"],row["xc"],row["engine"]))
    for cfg in ordered:
        cfg["process_limit_seconds"] = cfg["single_call_limit_seconds"]*cfg["repeats"]+120
        (inputs/(cfg["case_id"]+".json")).write_text(json.dumps(cfg,indent=2)+"\n")
    plan = dict(schema_version=1, catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        primary_systems=60, primary_basis_counts=fixtures["primary_basis_counts"],
        atom_range=fixtures["atom_range"], covered_elements=fixtures["covered_elements"],
        primary_ks_xc_counts=dict(Counter(f["family"] for _,_,f,s,_ in definitions if s=="primary")),
        independent_worker_count=len(ordered), max_logical_cpus=64, max_memory_bytes=256*1024**3,
        rss_guard_bytes=200*1024**3, single_calculation_limit_seconds=6000,
        protocol="60 primary systems balanced across five bases and four XC families; selected cross-controls only",
        representations="R/U up to128AO: conventional packed; largerR/U: explicitweigendRI; RSHR/U: direct; GHF/GKS: packed only",
        precision="CPUfloat64,DFTgrid3; same3-part convergence callback inPySCF, includingextra-cyclecheck",
        timing="Fresh process per configuration; primaryonecall, crosscontrolscold+3new-objectwarm; basispreflight disclosed",
        run_order=[cfg["case_id"] for cfg in ordered])
    (args.output/"run_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    print(json.dumps({key:plan[key] for key in ("primary_systems","independent_worker_count","primary_ks_xc_counts")}))


if __name__ == "__main__":
    main()
