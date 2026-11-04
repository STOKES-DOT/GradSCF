"""Performance acceptance/recovery never promotes an incomplete SCF process."""
import importlib.util
import json
import os
from pathlib import Path

import pytest


def campaign_module():
    path = Path(os.getenv("SCF_HARNESS_PATH", str(Path(__file__).parents[1]/
        "benchmarks/probes/scf_solver/campaign.py")))
    spec = importlib.util.spec_from_file_location("scf_campaign_records",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finish():
    return campaign_module().finish_attempt


def attempt(tmp_path, *, terminal="ok", count=4, process_status="ok", code=0, with_process=True):
    case = "example"
    (tmp_path/case).mkdir()
    (tmp_path/case/"summary.json").write_text(json.dumps(dict(status=terminal,scientific_pass=terminal=="ok")))
    (tmp_path/case/"measurements.jsonl").write_text(
        "".join(json.dumps(dict(scientific_pass=True,repeat=i))+"\n" for i in range(count)))
    if with_process:
        (tmp_path/(case+".process.json")).write_text(json.dumps(
            dict(status=process_status,returncode=code,wall_seconds=1.)))
    return finish()(tmp_path,case,dict(repeats=4),"123",Path("source"))


@pytest.mark.parametrize("terminal", ["error","resource_limited","unsupported"])
def test_later_failure_cannot_accept_an_earlier_successful_repeat(tmp_path, terminal):
    result = attempt(tmp_path,terminal=terminal,count=1)
    assert result["attempt_finished"]
    assert not result["run_complete"]
    assert not result["accepted_for_comparison"]
    assert result["executed_repeats"] == 1


def test_process_failure_overrides_old_success_summary(tmp_path):
    result = attempt(tmp_path,count=1,process_status="error",code=-9)
    assert result["status"] == "error"
    assert not result["accepted_for_comparison"]


def test_completed_guard_recovers_missing_derived_run_record(tmp_path):
    result = attempt(tmp_path)
    assert result["run_complete"] and result["accepted_for_comparison"]
    assert json.loads((tmp_path/"example.run.json").read_text()) == result


def test_orphan_attempt_preserves_evidence_without_claiming_completion(tmp_path):
    result = attempt(tmp_path,count=1,with_process=False)
    assert result["status"] == "interrupted_or_running"
    assert not result["run_complete"] and not result["accepted_for_comparison"]
    assert result["executed_repeats"] == 1


def test_cross_allocation_resume_keeps_original_worker_identity(tmp_path):
    attempt(tmp_path)
    (tmp_path/"example/runtime.json").write_text(json.dumps(dict(job="111",source_snapshot="original-source")))
    result = finish()(tmp_path,"example",dict(repeats=4),"222",Path("new-source"))
    assert result["allocation"] == "111"
    assert result["source_snapshot"] == "original-source"
    assert result["derived_under_allocation"] == "222"


@pytest.mark.parametrize("failure_kind", ["returncode", "exception"])
def test_launch_failure_preserves_logical_cwd_and_stops_campaign(tmp_path, monkeypatch, failure_kind):
    campaign = campaign_module()
    storage = tmp_path/"physical-campaign"
    storage.mkdir()
    logical = tmp_path/"logical-campaign"
    logical.symlink_to(storage, target_is_directory=True)
    (storage/"source").mkdir()
    plan = storage/"plan"
    (plan/"configs").mkdir(parents=True)
    cases = ["first", "second"]
    (plan/"run_plan.json").write_text(json.dumps(dict(run_order=cases, rss_guard_bytes=1024)))
    for case in cases:
        (plan/"configs"/(case+".json")).write_text(
            json.dumps(dict(repeats=1, threads=8, process_limit_seconds=60)))
    monkeypatch.chdir(storage)
    monkeypatch.setenv("PWD", str(tmp_path/"wrong-directory"))
    monkeypatch.setattr("sys.argv", [str(campaign.__file__), str(logical),
        str(logical/"source"), str(logical/"plan"), "123"])
    end = campaign.datetime.datetime.now()+campaign.datetime.timedelta(days=1)
    launches = []

    def fake_squeue(command, **kwargs):
        assert command[0] == "squeue"
        return end.isoformat()+"\n"

    def fake_srun(command, *, env, **kwargs):
        assert command[0] == "srun"
        launches.append(dict(command=command, env=env,
            cwd=Path(kwargs.get("cwd", os.getcwd())).resolve()))
        if failure_kind == "exception":
            raise OSError("synthetic srun launch failure")
        return campaign.subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(campaign.subprocess, "check_output", fake_squeue)
    monkeypatch.setattr(campaign.subprocess, "run", fake_srun)
    if failure_kind == "exception":
        with pytest.raises(OSError, match="synthetic srun launch failure"):
            campaign.main()
    else:
        campaign.main()

    assert launches[0]["env"]["PWD"] == str(logical)
    assert launches[0]["cwd"] == storage.resolve()
    assert len(launches) == 1
    output = storage/"results/final-matrix"
    if failure_kind == "returncode":
        assert json.loads((output/"uncertain_attempts.json").read_text()) == [cases[0]]
    assert not (storage/"release.123").exists()
    assert (output/(cases[0]+".log")).exists()
    assert not (output/(cases[0]+".process.json")).exists()
    assert not (output/(cases[1]+".log")).exists()
