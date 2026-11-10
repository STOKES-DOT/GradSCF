"""Tabulate completed SCF attempts without treating finite energies as convergence."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overlay", type=Path, help="Explicit correction attempts; original records remain untouched")
    parser.add_argument("--timing-exclusions", type=Path, help="Case IDs measured alongside diagnostics")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plan = json.loads((args.plan/"run_plan.json").read_text())
    excluded = set() if args.timing_exclusions is None else set(json.loads(args.timing_exclusions.read_text()))
    rows, pairs, record_sources = [], {}, {}
    for case in plan["run_order"]:
        cfg = json.loads((args.plan/"configs"/(case+".json")).read_text())
        directory = (args.overlay if args.overlay is not None and
            (args.overlay/(case+".run.json")).exists() else args.results)
        run_path = directory/(case+".run.json")
        run = json.loads(run_path.read_text()) if run_path.exists() else {}
        record_path = directory/case/"measurements.jsonl"
        record_sources[case] = record_path
        records = [] if not record_path.exists() else [json.loads(x) for x in record_path.read_text().splitlines()]
        latest = run.get("last_completed_scf",{})
        warm = [r["scf_facade_seconds"] for r in records[1:] if r.get("scientific_pass")]
        row = dict(case_id=case,system_id=cfg["id"],engine=cfg["engine"],method=cfg["method"],
            basis=cfg["basis"],xc=cfg["xc"],xc_family=cfg["xc_family"],natom=cfg["natom"],
            source_snapshot=run.get("source_snapshot"),allocation=run.get("allocation"),
            record_directory=str(directory/case),
            supersedes_run=str(args.results/(case+".run.json")) if directory != args.results else None,
            timing_eligible_for_comparison=case not in excluded,
            scopes=";".join(cfg["scopes"]),nao=latest.get("nao"),ngrid=latest.get("ngrid"),naux=latest.get("naux"),
            representation=latest.get("representation"),status=run.get("status","not_run"),
            execution_strategy_actual=latest.get("execution_strategy_actual"),
            approximation=latest.get("approximation"),
            converged=latest.get("converged"),scientific_pass=latest.get("scientific_pass",False),
            process_status=run.get("process_status"),run_complete=run.get("run_complete",False),
            attempt_finished=run.get("attempt_finished",False),
            accepted_for_comparison=run.get("accepted_for_comparison",False),
            repeats_executed=run.get("executed_repeats",0),repeats_requested=cfg["repeats"],
            energy_hartree=latest.get("energy_hartree"),gradient_norm=latest.get("gradient_norm"),
            cycles=latest.get("cycles"),orthogonality_error=latest.get("orthogonality_error"),
            first_scf_seconds=None if not records else records[0].get("scf_facade_seconds"),
            warm_scf_median_seconds=None if not warm else statistics.median(warm),
            warm_sample_count=len(warm),process_wall_seconds=run.get("process_wall_seconds"),
            peak_rss_bytes=run.get("process_rss_peak_bytes"),notes=latest.get("notes"),
            absolute_energy_difference_hartree=None,time_ratio_cold=None,time_ratio_warm=None,
            time_ratio_process=None,
            comparison_status="pending_or_unaccepted")
        rows.append(row)
        pairs.setdefault((cfg["id"],cfg["basis"],cfg["method"],cfg["xc"]),{})[cfg["engine"]] = row
    for paired in pairs.values():
        if set(paired) != {"gradscf","pyscf"}:
            continue
        g,p = paired["gradscf"],paired["pyscf"]
        matched = (g["accepted_for_comparison"] and p["accepted_for_comparison"]
                   and g["nao"]==p["nao"] and g["approximation"]==p["approximation"]
                   and g["naux"]==p["naux"])
        error = None
        if g["energy_hartree"] is not None and p["energy_hartree"] is not None:
            error = abs(g["energy_hartree"]-p["energy_hartree"])
        status = ("not_comparable" if not matched else "energy_or_state_mismatch"
                  if error is None or error > 1e-7 else "matched"
                  if g["execution_strategy_actual"]==p["execution_strategy_actual"] else "matched_task_different_strategy")
        cold = warm = None
        process_ratio = None
        if (status in ("matched","matched_task_different_strategy")
                and g["timing_eligible_for_comparison"] and p["timing_eligible_for_comparison"]):
            cold = g["first_scf_seconds"]/p["first_scf_seconds"]
            if g["warm_sample_count"] == p["warm_sample_count"] == 3:
                warm = g["warm_scf_median_seconds"]/p["warm_scf_median_seconds"]
            if g["repeats_requested"]==p["repeats_requested"]==1:
                process_ratio = g["process_wall_seconds"]/p["process_wall_seconds"]
        for row in (g,p):
            row.update(absolute_energy_difference_hartree=error,comparison_status=status,
                       time_ratio_cold=cold,time_ratio_warm=warm,time_ratio_process=process_ratio)
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output/"measurements.jsonl").open("w") as stream:
        for case in plan["run_order"]:
            source = record_sources[case]
            if source.exists():
                stream.write(source.read_text())
    counts = dict(planned_configurations=len(rows),statuses=dict(Counter(r["status"] for r in rows)),
        completed_configurations=sum(r["run_complete"] for r in rows),
        attempted_configurations=sum(r["attempt_finished"] for r in rows),
        scientifically_accepted_configurations=sum(r["accepted_for_comparison"] for r in rows),
        completed_primary_systems=len({r["system_id"] for r in rows if "primary" in r["scopes"] and r["run_complete"]}),
        interpretation="A system counted here has at least one completed configuration, not necessarily all six methods.",
        comparison_statuses=dict(Counter(r["comparison_status"] for r in rows if r["engine"]=="gradscf")),
        units="Hartree,seconds,bytes; ratiosGradSCF/reference; no mixed-method average")
    (args.output/"counts.json").write_text(json.dumps(counts,indent=2)+"\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
