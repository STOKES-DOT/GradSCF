#!/usr/bin/env bash
# Four additional standard bases, with contracted ERI <= 2.34 GiB each.
set -euo pipefail
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL PYTHONPATH=src
py=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
work=artifacts/aniline-basis-comparison-20260920
taskset -c 0-3 "$py" tools/compare_methane_core6_basis_sets.py "$work/comparison-input.json" --output "$work/remote-polarized" --max-eri-gib 3 --bases '6-31g(d)' '6-31g(d,p)' def2-svp cc-pvdz > "$work/remote-polarized.log" 2>&1
taskset -c 0-3 "$py" tests/comparisons/validate_methane_core6_basis_sets.py "$work/remote-polarized/comparison.json" "$work/comparison-input.json" > "$work/remote-polarized-reference.log" 2>&1
