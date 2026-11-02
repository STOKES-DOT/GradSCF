#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-core-smoke
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/nnao-core-benchmarks-20261020/smoke-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax:$PWD/tests/comparisons
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-core-benchmarks-20261020/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -u tests/comparisons/nnao_core/run.py \
  --dataset artifacts/nnao-core-benchmarks-20261020/dataset.json \
  --output artifacts/nnao-core-benchmarks-20261020/smoke \
  --checkpoint artifacts/nnao-relative500-20261018/checkpoint-current.npz \
  --snapshot artifacts/nnao-relative500-20261018/snapshot.json \
  --profile smoke
