#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=shared-SCF-response
#SBATCH --nodelist=n006
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:NV5090:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
RUN=/home/user_yuanjiao/private/homefile/gradscf-shared-execution-20261110
export RUN PY=/home/user_yuanjiao/private/homefile/gradscf-libcint-nnao-20261026/env-compare/bin/python
cd "$RUN"
export PYTHONPATH="$RUN/src:$RUN:$RUN/tests" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export JAX_ENABLE_COMPILATION_CACHE=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p results
scontrol show job "$SLURM_JOB_ID" > results/allocation.txt
lscpu > results/lscpu.txt
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv > results/gpu.csv
idle=$(date +%s)
while ! test -e STOP; do
 did=0
 for task in queue/*.sh; do
  test -e "$task.ready" && test ! -e "$task.exit" || continue
  did=1
  printf 'START %s %s\n' "$task" "$(date -Is)"
  set +e
  bash "$task" > "$task.log" 2>&1
  code=$?
  set -e
  printf '%s\n' "$code" > "$task.exit"
  printf 'END %s code=%s %s\n' "$task" "$code" "$(date -Is)"
  idle=$(date +%s)
 done
 if ((did==0));then
  (($(date +%s)-idle<900)) || break
  sleep 5
 fi
done
printf 'END %s\n' "$(date -Is)"
