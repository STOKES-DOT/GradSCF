"""Record whitelisted environment metadata against an actual run manifest."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
run = json.loads(args.manifest.read_text())
library = Path(run["native_path"])
assert hashlib.sha256(library.read_bytes()).hexdigest() == run["native_sha256"]
versions = {}
for package in ("jax", "jaxlib", "numpy", "scipy", "jax-xc", "pyscf", "optax", "jax-cuda12-plugin"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
keys = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
        "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")
profiler = Path("profiling-pyspy-0.4.1/bin/py-spy")
payload = dict(python=sys.version, executable=sys.executable, platform=platform.uname()._asdict(),
    versions=versions, manifest=str(args.manifest), manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    actual_run=run, metadata_process_affinity=sorted(os.sched_getaffinity(0)),
    allocation={key: os.getenv(key) for key in keys},
    profiler=dict(version="0.4.1", isolated_target=str(profiler.parent.parent),
        binary_sha256=hashlib.sha256(profiler.read_bytes()).hexdigest()) if profiler.exists() else None,
    notes="Metadata only; actual numeric affinity/device/thread/source are recorded in the run records. No full environment or credentials captured.")
args.output.write_text(json.dumps(payload, indent=2)+"\n")
print("Environment and actual native library hash recorded", flush=True)
