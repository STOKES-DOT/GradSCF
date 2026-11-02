#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-dimer4-finish
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014
#SBATCH --output=artifacts/nnao-dimer-controls-20261026/finish-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/tests/comparisons
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-dimer-controls-20261026/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
date -Is
if python -c 'import json,sys; x=json.load(open("artifacts/nnao-dimer-controls-20261026/run/results.json")); sys.exit(0 if x["status"]=="complete" and x["passed"]==204 and x["failed"]==0 else 1)'; then
    echo 'All 204 cases already passed; no SCF evaluations required.'
    exit 0
fi
python -c 'import json,jax; x=json.load(open("artifacts/nnao-dimer-controls-20261026/run/results.json")); assert x["failed"]==0,"Review failed numerical cases before resuming"; assert len(jax.devices("gpu"))==1; g=json.load(open("artifacts/nnao-dimer-controls-20261026/run/gate.json")); assert g["status"]=="passed"'
python artifacts/nnao-dimer-controls-20261026/measure.py \
  --output artifacts/nnao-dimer-controls-20261026/run/resource-${SLURM_JOB_ID}.json \
  -- python -u tests/comparisons/nnao_dimer_controls/run.py \
  --inputs artifacts/nnao-dimer-controls-20261026/inputs/cases.json \
  --output artifacts/nnao-dimer-controls-20261026/run
date -Is
