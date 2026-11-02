#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-ethane-extra-p
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/nnao-ethane-p2-20261018/run-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-ethane-p2-20261018/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -m pytest -q artifacts/nnao-ethane-p2-20261018/test_expansion.py
python -u artifacts/nnao-ethane-p2-20261018/run.py
python -u artifacts/nnao-ethane-p2-20261018/gpu4_check.py
