"""Verify frozen inputs, preserve checkpoints and prepare graph-off Slurm files."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
REPOSITORY = CAMPAIGN.parents[1]
LOGICAL_REPOSITORY = Path("/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021")
LANES = {"szp": ("s", "m"), "dzp": ("s", "m", "l"), "tzp": ("s", "m", "l")}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def checkpoint_info(path):
    with zipfile.ZipFile(path) as archive, archive.open("metadata.npy") as stream:
        metadata = json.loads(str(np.lib.format.read_array(stream, allow_pickle=False).item()))
    result = {key: metadata.get(key) for key in
              ("epoch", "update_step", "batch_cursor", "online_sample_count", "gradient_update_step")}
    result["identity_sha256"] = hashlib.sha256(
        json.dumps(metadata["identity"], sort_keys=True).encode()).hexdigest()
    result["scientific_source_sha256"] = metadata["identity"]["scientific_source_sha256"]
    result["sha256"] = file_hash(path)
    result["path"] = str(path)
    return result


def main():
    assert not (HERE / "plan.json").exists(), "Never silently replace a prepared restart."
    config = json.loads((CAMPAIGN / "campaign.json").read_text())
    assert LOGICAL_REPOSITORY.resolve() == REPOSITORY
    digest = hashlib.sha256()
    package = REPOSITORY / "src/gradscf"
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".dat"}:
            digest.update(str(path.relative_to(package)).encode())
            digest.update(file_hash(path).encode())
    assert digest.hexdigest() == config["scientific_source_sha256"], "Frozen scientific source changed."
    for relative, expected in dict(config["input_sha256"], **config["workflow_sha256"]).items():
        assert file_hash(CAMPAIGN / relative) == expected, relative
    ready = json.loads((CAMPAIGN / "ready.json").read_text())
    assert ready["status"] == "complete"
    for tier in ("s", "m", "l"):
        assert ready["candidate_sha256"][tier] == file_hash(CAMPAIGN / f"inputs/{tier}-candidate.json")
    old_outputs = {}
    for lane, tiers in LANES.items():
        for tier in tiers:
            family = f"nnao_{lane}_{tier}"
            output = CAMPAIGN / "runs" / family / "train"
            snapshot = HERE / "before" / family
            for name in ("progress.json", "phase.json", "batch_progress.json", "history.json", "metadata.json"):
                source = output / name
                if source.exists():
                    snapshot.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, snapshot / name)
            checkpoint = output / "last.npz"
            if checkpoint.exists():
                info = checkpoint_info(checkpoint)
                assert info["scientific_source_sha256"] == config["scientific_source_sha256"]
                assert info["update_step"] == 16, (family, info)
                snapshot.mkdir(parents=True, exist_ok=True)
                # Checkpoint saving uses os.replace, so this link preserves the old inode.
                os.link(checkpoint, snapshot / "last.npz")
                old_outputs[family] = info
            else:
                old_outputs[family] = dict(update_step=0, epoch=0, batch_cursor=0, mode="original_seed")
    # Promote only the successfully completed, identity-compatible S-TZP checkpoint.
    diagnostic = CAMPAIGN / "diagnostics/cuda-graph-off-20261022"
    result = json.loads((diagnostic / "result.json").read_text())
    seed = diagnostic / "run/last.npz"
    info = checkpoint_info(seed)
    assert result["status"] == "passed_update_budget" and info["update_step"] == 40
    assert info["sha256"] == result["checkpoint_sha256"]
    assert info["identity_sha256"] == old_outputs["nnao_tzp_s"]["identity_sha256"]
    write(HERE / "diagnostic-seed.json", info)
    target = CAMPAIGN / "runs/nnao_tzp_s/train/last.npz"
    temporary = target.with_name("graph-off-update40.npz")
    os.link(seed, temporary)
    os.replace(temporary, target)
    old_outputs["nnao_tzp_s"] = dict(info, mode="resume_completed_graph_off_control")
    resource_policy = {
        "szp": dict(gpu="A800", node="n005", memory_gib=448, time_limit_hours=8),
        "dzp": dict(gpu="A100_80G", node=None, memory_gib=512, time_limit_hours=72),
        "tzp": dict(gpu="A800", node="n005", memory_gib=512, time_limit_hours=8),
    }
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),
                scientific_source_sha256=config["scientific_source_sha256"],
                runtime_change="--xla_gpu_enable_command_buffer= before JAX initialization",
                max_simultaneous_gpus=3, cpus_per_task=32,
                automatic_requeue="USR1 five minutes before each allocation time limit",
                epochs=1000, batch_size=50, learning_rate=1e-3, decay_epochs=100,
                seed=0, early_stop_window=10, early_stop_tolerance_hartree=1e-4,
                finite_differences=False, extra_evaluations=False,
                lanes={lane: [f"nnao_{lane}_{tier}" for tier in tiers] for lane, tiers in LANES.items()},
                resource_policy=resource_policy, dependency_policy="afterok",
                families=old_outputs, original_source_and_inputs_changed=False)
    write(HERE / "plan.json", plan)
    render_scripts(plan)
    write(HERE / "preflight.json", dict(status="passed", source_hash=digest.hexdigest(),
          original_workflow_checks="passed", ready_gate="complete", slurm_syntax_files=8,
          checkpoint_states=old_outputs, promoted_checkpoint_sha256=info["sha256"],
          numerical_recalculation=False))
    print(json.dumps(plan, indent=2))


def render_scripts(plan):
    for lane, families in plan["lanes"].items():
        policy = plan["resource_policy"][lane]
        for family in families:
            text = f'''#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name={family}_goff
#SBATCH --gres=gpu:{policy["gpu"]}:1
#SBATCH --cpus-per-task=32
#SBATCH --mem={policy["memory_gib"]}G
#SBATCH --time={policy["time_limit_hours"]:02d}:00:00
#SBATCH --chdir={LOGICAL_REPOSITORY}
#SBATCH --output={HERE.relative_to(REPOSITORY)}/{family}-%j.log
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
'''
            text += f'#SBATCH --nodelist={policy["node"]}\n' if policy["node"] else '#SBATCH --exclude=n005\n'
            text += f'''set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${{XLA_FLAGS:-}} --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-family500-20261021/jit-cache/{family}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
date -Is
requeue_for_time_limit() {{
    trap - USR1
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}}
trap requeue_for_time_limit USR1
python -u {HERE}/run_family.py --family {family} &
worker=$!
wait "$worker"
trap - USR1
date -Is
'''
            path = HERE / (family + ".slurm")
            path.write_text(text)
            subprocess.run(["bash", "-n", str(path)], check=True)


if __name__ == "__main__":
    main()
