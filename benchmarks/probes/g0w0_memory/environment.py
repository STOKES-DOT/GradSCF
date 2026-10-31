"""Record the actually loaded isolated sources/library inside the allocation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    import jax
    import gradscf
    from gradscf.integrals._native import register_integrals
    handle = register_integrals()
    library = Path(handle._name).resolve()
    model = next((line.split(":", 1)[1].strip() for line in
        Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None)
    expected = json.loads(Path(sys.argv[1]).read_text())
    source = Path(gradscf.__file__).resolve().parents[2]
    mismatches = [p for p, digest in expected.items() if not (source/p).is_file()
                  or hashlib.sha256((source/p).read_bytes()).hexdigest() != digest]
    versions = {}
    for name in ("jax", "jaxlib", "numpy", "scipy", "pyscf", "jax-xc", "optax", "flax"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(python=sys.executable, gradscf_import=gradscf.__file__,
        checked_source_files=len(expected), source_mismatches=mismatches,
        native_loaded_path=str(library), native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        native_maps=sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if "libgradscf_integrals" in line}),
        cpu_model=model, affinity=sorted(os.sched_getaffinity(0)),
        versions=versions, devices=[dict(device=str(d), kind=d.device_kind) for d in jax.devices()],
        job_id=os.getenv("SLURM_JOB_ID"), node=os.uname().nodename,
        thread_environment={name: os.getenv(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GRADSCF_NATIVE_THREADS")},
        compilation_cache_enabled=bool(jax.config.jax_enable_compilation_cache))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2)+"\n")
    assert not mismatches, mismatches


if __name__ == "__main__":
    main()
