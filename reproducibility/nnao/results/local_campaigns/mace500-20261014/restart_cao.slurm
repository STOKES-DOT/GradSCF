#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao500-mace256-b50
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/mace500-20261014/restart-cao-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/mace500-20261014/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -u artifacts/mace500-20261014/restart_cao.py
