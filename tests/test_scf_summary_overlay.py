"""A corrected attempt is explicit and never overwrites the original evidence."""
import csv
import importlib.util
import json
from pathlib import Path
import sys


def test_explicit_overlay_selects_corrected_attempt(tmp_path, monkeypatch):
    path = Path(__file__).parents[1]/"benchmarks/probes/scf_solver/summarize.py"
    spec = importlib.util.spec_from_file_location("scf_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan, original, correction, output = [tmp_path/name for name in ("plan", "original", "correction", "out")]
    (plan/"configs").mkdir(parents=True)
    original.mkdir()
    correction.mkdir()
    case = "example"
    (plan/"run_plan.json").write_text(json.dumps(dict(run_order=[case])))
    cfg = dict(id="fixture", engine="gradscf", method="GHF", basis="sto-3g", xc="hf",
               xc_family="HF", natom=5, scopes=["primary"], repeats=1)
    (plan/"configs"/(case+".json")).write_text(json.dumps(cfg))
    old = original/(case+".run.json")
    old.write_text(json.dumps(dict(status="error", source_snapshot="old")))
    old_bytes = old.read_bytes()
    (correction/(case+".run.json")).write_text(json.dumps(dict(status="ok", source_snapshot="corrected",
        run_complete=True, attempt_finished=True, accepted_for_comparison=True,
        last_completed_scf=dict(scientific_pass=True, energy_hartree=-1.))))
    (correction/case).mkdir()
    (correction/case/"measurements.jsonl").write_text(json.dumps(dict(scientific_pass=True, scf_facade_seconds=1.))+"\n")
    exclusions = tmp_path/"timing-exclusions.json"
    exclusions.write_text(json.dumps([case]))
    monkeypatch.setattr(sys, "argv", [str(path), str(plan), str(original), str(output),
        "--overlay", str(correction), "--timing-exclusions", str(exclusions)])
    module.main()
    row = next(csv.DictReader((output/"summary.csv").open()))
    assert row["status"] == "ok" and row["source_snapshot"] == "corrected"
    assert row["supersedes_run"] == str(old)
    assert row["timing_eligible_for_comparison"] == "False"
    assert row["time_ratio_cold"] == ""
    assert (output/"measurements.jsonl").read_text() == (correction/case/"measurements.jsonl").read_text()
    assert old.read_bytes() == old_bytes
