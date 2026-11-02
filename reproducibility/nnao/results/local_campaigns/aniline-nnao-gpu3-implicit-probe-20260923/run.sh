#!/bin/bash
set -euo pipefail
root=/home/yjiao/GradSCF
work="$root/artifacts/aniline-nnao-gpu3-implicit-probe-20260923"
smoke="$root/artifacts/aniline-nnao-gpu3-smoke-20260923"
python=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
mkdir -p "$work/output"
cd "$root"
export PYTHONPATH="$root/src"
export JAX_ENABLE_X64=1 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMEXPR_NUM_THREADS=64
taskset -c 0-63 env CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$python" -u artifacts/aniline-basis-comparison-20260920/runner.py optimize \
  --geometry "$root/artifacts/eri-fix-20260920/geometries/aniline.json" \
  --basis-family szp663_direct --jk-backend df --auxbasis cc-pvqz-jkfit \
  --integral-cache "$smoke/aniline-szp663-ccpvqzjkfit.npz" \
  --initial-checkpoint "$smoke/training/checkpoint.npz" \
  --maxiter 0 --output-dir "$work/output" >"$work/probe.log" 2>&1
