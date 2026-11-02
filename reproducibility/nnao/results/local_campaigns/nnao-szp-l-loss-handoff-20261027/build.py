"""Prepare the dedicated L-SZP loss handoff without changing old archives."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "nnao-mace500-loss-20261019"
PAYLOAD = OUT / "payload"
REMOTE = "/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014/artifacts/mace500-20261014"
REMOTE_HASHES = {
    "train/history.json": "1d58f3b6727f4ab939372d479ca9bb9c8adecc33761b487b40d1cbd05505b6b6",
    "train/progress.json": "b4d878f583370c84557cf3df95f12fe70983ff23f68b6a297b54427b602a49b9",
    "inputs/model-config.json": "724a35ec30635435a75a65554baa77f657f4012475e82ff5cc6329bb0da84c3b",
    "single-pass-switch.json": "5d7a622dc6856ecceab71cb9e0c48bc13c35c189bbdd3a50fbb9bba27e872b30",
    "restart-single-pass-99005.log": "e2da5e5c8cd3f2f2cbdeed142a4c5a4976fa9bf7f0cc4294c51ab6482fdff2a2",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    PAYLOAD.mkdir(exist_ok=True)
    files = []
    for source in sorted((SOURCE / "snapshot").rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(SOURCE / "snapshot")
        if str(relative) in REMOTE_HASHES:
            assert sha(source) == REMOTE_HASHES[str(relative)]
        target = PAYLOAD / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append(dict(file=str(target.relative_to(PAYLOAD)), sha256=sha(source), bytes=source.stat().st_size))
    assert all((PAYLOAD / "snapshot" / name).is_file() for name in REMOTE_HASHES)
    shutil.copy2(SOURCE / "plot_loss.py", PAYLOAD / "plot_loss.py")
    shutil.copy2(OUT / "README.md", PAYLOAD / "README.md")
    provenance = dict(repository_name="NNAO-SZP-L", user_label="NAOO-SVP-L",
        scope="Historical 500-system fixed-exponent MACE training, not the active L-TZP run",
        job_id=99005, source_host="jy_gpu_2", remote_root=REMOTE,
        remote_verified_at="2026-10-27T21:41:34+08:00", remote_sha256=REMOTE_HASHES,
        remote_metadata_sha256="2b1bd16ef705e58322469b932e540a214226d102fd7acec9ee8384fcf6be1e04",
        remote_metadata_included=False, reason="54 MB full metadata omitted; compact checkpoint position, model config, source and policy switch retained",
        raw_files=files, plot_source_sha256=sha(SOURCE / "plot_loss.py"),
        old_paper_case="data/computational_examples/2026-10-21/cases/11_training_loss_snapshot",
        dataset="data/training/nnao500-20261021", evaluation_model_update=3351,
        evaluation_model_parameter_sha256="2f4386c12cb4637b0b1f92682c86dba2b298b1e9c27f1926f1c2e24a1f6f1ade",
        source_status="TIME_LIMIT", stop_time="2026-10-19T17:10:22+08:00",
        last_complete_epoch=596, saved_update=5969, partial_epoch=597, partial_batches=9,
        new_scf_calls=0, training_state_modified=False)
    (PAYLOAD / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(status="prepared", raw_files=len(files), destination=str(PAYLOAD)), indent=2))


if __name__ == "__main__":
    main()
