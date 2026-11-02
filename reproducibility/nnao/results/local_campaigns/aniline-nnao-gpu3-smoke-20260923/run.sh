#!/bin/bash
set -euo pipefail

root=/home/yjiao/GradSCF
work="$root/artifacts/aniline-nnao-gpu3-smoke-20260923"
python=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
geometry="$root/artifacts/eri-fix-20260920/geometries/aniline.json"
cache="$work/aniline-szp663-ccpvqzjkfit.npz"

mkdir -p "$work"
cd "$root"

export PYTHONPATH="$root/src"
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

echo "cache_started=$(date -Is)" | tee "$work/status.log"
taskset -c 0-63 env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  "$python" -u tools/optimize_methane_nnao.py \
  --geometry "$geometry" \
  --basis-family szp663_direct \
  --jk-backend df \
  --auxbasis cc-pvqz-jkfit \
  --prepare-integral-cache "$cache" \
  >"$work/cache.log" 2>&1
echo "cache_finished=$(date -Is)" | tee -a "$work/status.log"

compute_pids=$(nvidia-smi --id=3 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
if [ "$compute_pids" -ne 0 ]; then
  echo "training_aborted=$(date -Is) reason=gpu3-became-busy compute_pids=$compute_pids" | tee -a "$work/status.log"
  exit 75
fi

echo "training_started=$(date -Is) gpu=3" | tee -a "$work/status.log"
taskset -c 0-63 env CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$python" -u artifacts/aniline-basis-comparison-20260920/runner.py optimize \
  --geometry "$geometry" \
  --basis-family szp663_direct \
  --jk-backend df \
  --auxbasis cc-pvqz-jkfit \
  --integral-cache "$cache" \
  --maxiter 1 \
  --output-dir "$work/training" \
  >"$work/training.log" 2>&1
echo "training_finished=$(date -Is)" | tee -a "$work/status.log"
