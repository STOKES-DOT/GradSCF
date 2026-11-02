#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-pool-gpu-check
#SBATCH --gres=gpu:A100_40G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/pool-calibration-repair-20261014/validate-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/pool-calibration-repair-20261014/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PLATFORMS=cpu
python -m pytest -q tests/nnao/test_pool_devices.py tests/nnao/test_pool_objective.py tests/nnao/test_pool_workflow.py tests/nnao/test_basis_family_grid.py tests/nnao/test_fixed_pool_training.py
export JAX_PLATFORMS=cuda,cpu
python -m pytest -q tests/nnao/test_pool_devices.py
for family in nnao_dzp_s nnao_tzp_s; do
  python -u artifacts/pool-calibration-repair-20261014/validate_calibration.py "$family"
done
for zeta in szp dzp tzp; do
  for tier in s m l; do
    python -u artifacts/pool-calibration-repair-20261014/replay_training.py nnao_${zeta}_${tier}
  done
done
python -u artifacts/pool-calibration-repair-20261014/deploy.py
