#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao200-stability443
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/nnao-resume443-stability-20261013/restart-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-resume443-stability-20261013/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PLATFORMS=cpu
python -m pytest -q tests/test_df_rhf_stability.py tests/test_df_rhf_stability_curvature.py
export JAX_PLATFORMS=cuda,cpu
python -u artifacts/nnao-resume443-stability-20261013/diagnose.py baseline
python -u artifacts/nnao-resume443-stability-20261013/diagnose.py fixed
python -u artifacts/nnao-resume443-stability-20261013/restart.py
