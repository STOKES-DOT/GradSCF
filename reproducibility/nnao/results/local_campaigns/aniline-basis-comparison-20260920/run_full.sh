#!/usr/bin/env bash
# Prepared job, not launched. Run from the isolated current-source snapshot.
set -euo pipefail
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL
export PYTHONPATH=src
py=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
work=artifacts/aniline-basis-comparison-20260920
# 700 GiB process virtual-memory ceiling; no unrelated jobs are stopped.
ulimit -v 734003200
"$py" - <<'PY'
from pathlib import Path
available=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
assert available > 750*1024**2, 'Less than 750 GiB available; defer this high-memory job.'
PY
mkdir -p "$work/full"
# Each process releases its allocations before the next basis family starts.
/usr/bin/time -v taskset -c 0-7 "$py" tools/compare_methane_core6_basis_sets.py "$work/comparison-input.json" --output "$work/full/standard" --max-eri-gib 80 --bases sto-3g 3-21g 6-31g '6-31g(d)' '6-31g(d,p)' def2-svp cc-pvdz def2-tzvp cc-pvtz > "$work/full/standard.log" 2>&1
for family in szp442_direct szp663_direct; do
    target="$work/full/$family"
    mkdir -p "$target"
    # Both start from the same embedded reference contractions. No methane
    # coefficients are assigned to chemically inequivalent aniline atoms.
    /usr/bin/time -v taskset -c 0-7 "$py" "$work/runner.py" optimize --geometry "$work/geometry.json" --basis-family "$family" --max-primitive-eri-gib 220 --maxiter 600 --output-dir "$target" > "$target/run.log" 2>&1
    taskset -c 0-7 "$py" tests/comparisons/validate_methane_nnao.py "$target/summary.json" > "$target/reference.log" 2>&1
    if [ "$family" = szp442_direct ]; then label=NNAO-core6-4s4p2d; else label=NNAO-core6-6s6p3d-H6s2p; fi
    taskset -c 0-7 "$py" tools/compare_methane_core6_basis_sets.py "$target/summary.json" --bases "$label" --output "$target/comparison" > "$target/comparison.log" 2>&1
done
taskset -c 0-7 "$py" tests/comparisons/validate_methane_core6_basis_sets.py "$work/full/standard/comparison.json" "$work/comparison-input.json" > "$work/full/standard-reference.log" 2>&1
