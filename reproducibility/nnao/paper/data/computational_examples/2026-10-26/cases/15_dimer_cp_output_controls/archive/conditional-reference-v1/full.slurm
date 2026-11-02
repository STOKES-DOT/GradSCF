#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-dimer4-cp
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014
#SBATCH --output=artifacts/nnao-dimer-controls-20261026/full-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/tests/comparisons
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-dimer-controls-20261026/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
date -Is
python -c 'import json,jax; assert len(jax.devices("gpu"))==1; x=json.load(open("artifacts/nnao-dimer-controls-20261026/run/gate.json")); assert x["status"]=="passed"'
python artifacts/nnao-dimer-controls-20261026/measure.py \
  --output artifacts/nnao-dimer-controls-20261026/run/resource-${SLURM_JOB_ID}.json \
  -- python -u tests/comparisons/nnao_dimer_controls/run.py \
  --inputs artifacts/nnao-dimer-controls-20261026/inputs/cases.json \
  --output artifacts/nnao-dimer-controls-20261026/run
date -Is
