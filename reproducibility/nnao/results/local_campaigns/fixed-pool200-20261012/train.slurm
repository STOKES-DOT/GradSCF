#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-fixed200-fresh
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/fixed-pool200-20261012/slurm-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/fixed-pool200-20261012/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
sha256sum -c --quiet artifacts/fixed-pool200-20261012/source-sha256.txt
python -u artifacts/fixed-pool200-20261012/launch.py
