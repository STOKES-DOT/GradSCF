#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-fixed-perf-200
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/fixed-pool-performance-20261012/full200-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/artifacts/fixed-pool-performance-20261012/source/src:$PWD/artifacts/fixed-pool-performance-20261012/source/src/gradscf/model/nnao/vendor/mace-jax
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/fixed-pool-performance-20261012/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -u -m gradscf.model.nnao.workflows.fixed_pool \
  --platform cuda --seed 0 --learning-rate 1e-4 --epochs 1 --storage mmap \
  --references artifacts/fixed-pool200-20261012/references.json \
  --candidate artifacts/fixed-pool200-20261012/candidate-exponents.json \
  --cache-dir artifacts/fixed-pool200-20261012/cache \
  --output-dir artifacts/fixed-pool-performance-20261012/full200
