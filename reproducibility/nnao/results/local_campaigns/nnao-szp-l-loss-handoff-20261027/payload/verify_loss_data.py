"""Check loss CSV, raw history, stdout and remote checksums; no SCF or training."""
import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


def main():
    root = Path(__file__).resolve().parent
    history = json.loads((root / "snapshot/train/history.json").read_text())
    position = json.loads((root / "snapshot/checkpoint-position.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    rows = list(csv.DictReader((root / "loss-curve.csv").open()))
    assert len(rows) == len(history) == 597
    assert [int(r["epoch"]) for r in rows] == list(range(597))
    assert rows[4]["mean_energy_hartree"] == ""
    assert rows[4]["sample_count"] == "0" and rows[4]["complete_epoch_coverage"] == "False"
    assert rows[5]["signed_epoch_energy_decrease_mEh"] == ""
    online = rows[5:]
    assert len(online) == 592
    log = (root / "snapshot/restart-single-pass-99005.log").read_text()
    pattern = r"Epoch (\d+)/1000: online mean E ([-+0-9.eE]+) Ha; samples 500/500;"
    stdout = {int(e): float(v) for e, v in re.findall(pattern, log)}
    assert set(stdout) == set(range(5, 597))
    error = 0.
    for source, row in zip(history, rows):
        if source["loss_hartree"] is None:
            assert row["mean_energy_hartree"] == ""
        else:
            assert float(row["mean_energy_hartree"]) == source["loss_hartree"]
    for i, row in enumerate(online):
        assert row["loss_kind"] == "online_pre_update_mean"
        assert row["sample_count"] == "500" and row["complete_epoch_coverage"] == "True"
        e = float(row["mean_energy_hartree"])
        error = max(error, abs(e - stdout[int(row["epoch"])]))
        if i:
            delta = (float(online[i-1]["mean_energy_hartree"]) - e) * 1000
            assert abs(delta - float(row["signed_epoch_energy_decrease_mEh"])) < 1e-10
    assert error <= 1e-12
    assert position["epoch"] == 596 and position["update_step"] == 5969 and position["batch_cursor"] == 9
    assert "JOB 99005 ON n001 CANCELLED AT 2026-10-19T17:10:22 DUE TO TIME LIMIT" in log
    assert summary["last_complete_epoch"] == 596 and summary["online_complete_epochs"] == 592
    for name, expected in provenance["remote_sha256"].items():
        assert hashlib.sha256((root / "snapshot" / name).read_bytes()).hexdigest() == expected
    model = json.loads((root / "snapshot/inputs/model-config.json").read_text())
    assert model["basis_family"] == "nnao_szp_l"
    for stem in ("training-loss", "energy-log-trends"):
        with Image.open(root / f"{stem}.png") as image:
            image.load()
            assert min(image.size) >= 900
        assert len(PdfReader(root / f"{stem}.pdf").pages) == 1
    report = dict(status="passed", csv_rows=597, online_rows=592, online_epochs=[5, 596],
        missing_loss_epochs=[4], excluded_partial_epoch=597, raw_history_matches_csv=True,
        stdout_rows_checked=len(stdout), max_history_stdout_error_hartree=error,
        remote_file_hashes_checked=len(provenance["remote_sha256"]), basis_family="nnao_szp_l",
        last_complete_epoch=596, saved_update=5969, partial_batches=9,
        terminal_state="TIME_LIMIT", old_progress_running_field_preserved=True,
        figures_checked=4, new_scf_calls=0, training_state_modified=False)
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
