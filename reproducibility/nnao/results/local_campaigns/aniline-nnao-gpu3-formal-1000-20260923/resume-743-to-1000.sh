#!/bin/bash
set -euo pipefail
root=/home/yjiao/GradSCF
work="$root/artifacts/aniline-nnao-gpu3-formal-1000-20260923"
smoke="$root/artifacts/aniline-nnao-gpu3-smoke-20260923"
python=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
output="$work/training-resume-743-1000"
mkdir -p "$output"
cd "$root"
export PYTHONPATH="$root/src"
export JAX_ENABLE_X64=1 OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMEXPR_NUM_THREADS=64
compute_pids=$(nvidia-smi --id=3 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
if [ "$compute_pids" -ne 0 ]; then
  echo "resume_aborted=$(date -Is) reason=gpu3-busy compute_pids=$compute_pids" | tee "$work/resume-status.log"
  exit 75
fi
echo "resume_started=$(date -Is) gpu=3 remaining_maxiter=258" | tee "$work/resume-status.log"
taskset -c 0-63 env CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$python" -u artifacts/aniline-basis-comparison-20260920/runner.py optimize \
  --geometry "$root/artifacts/eri-fix-20260920/geometries/aniline.json" \
  --basis-family szp663_direct --jk-backend df --auxbasis cc-pvqz-jkfit \
  --integral-cache "$smoke/aniline-szp663-ccpvqzjkfit.npz" \
  --initial-checkpoint "$work/training/checkpoint-latest.npz" \
  --maxiter 258 --output-dir "$output" >"$work/resume.log" 2>&1
echo "resume_finished=$(date -Is)" | tee -a "$work/resume-status.log"
