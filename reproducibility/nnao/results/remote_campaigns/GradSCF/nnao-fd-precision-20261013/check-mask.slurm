#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-frozen-mask
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/nnao-fd-precision-20261013/mask-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/fixed-pool-performance-20261012/jit-cache
python -u artifacts/nnao-fd-precision-20261013/check_mask.py
