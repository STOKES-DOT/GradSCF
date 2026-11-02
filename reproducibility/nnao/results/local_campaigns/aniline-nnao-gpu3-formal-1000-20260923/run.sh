#!/bin/bash
set -euo pipefail

root=/home/yjiao/GradSCF
work="$root/artifacts/aniline-nnao-gpu3-formal-1000-20260923"
smoke="$root/artifacts/aniline-nnao-gpu3-smoke-20260923"
probe="$root/artifacts/aniline-nnao-gpu3-implicit-probe-20260923/output"
python=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python

mkdir -p "$work/training"
cd "$root"

export PYTHONPATH="$root/src"
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

compute_pids=$(nvidia-smi --id=3 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
if [ "$compute_pids" -ne 0 ]; then
  echo "launch_aborted=$(date -Is) reason=gpu3-busy compute_pids=$compute_pids" | tee "$work/status.log"
  exit 75
fi

echo "training_started=$(date -Is) gpu=3 maxiter=1000" | tee "$work/status.log"
sha256sum tools/optimize_methane_nnao.py "$work/run.sh" >"$work/source-sha256.txt"
nvidia-smi --id=3 --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader >"$work/gpu-at-launch.txt"

taskset -c 0-63 env CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$python" -u artifacts/aniline-basis-comparison-20260920/runner.py optimize \
  --geometry "$root/artifacts/eri-fix-20260920/geometries/aniline.json" \
  --basis-family szp663_direct \
  --jk-backend df \
  --auxbasis cc-pvqz-jkfit \
  --integral-cache "$smoke/aniline-szp663-ccpvqzjkfit.npz" \
  --initial-checkpoint "$probe/checkpoint.npz" \
  --maxiter 1000 \
  --output-dir "$work/training" \
  >"$work/training.log" 2>&1

echo "training_finished=$(date -Is)" | tee -a "$work/status.log"
