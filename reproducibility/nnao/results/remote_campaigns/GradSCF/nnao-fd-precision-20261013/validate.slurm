#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-mask-regression
#SBATCH --gres=gpu:A100_40G:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/nnao-fd-precision-20261013/validate-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
JAX_PLATFORMS=cpu python -m pytest -q tests/nnao/test_fixed_pool_training.py tests/nnao/test_joint_native_training.py tests/nnao/test_shared_pilot_training.py
