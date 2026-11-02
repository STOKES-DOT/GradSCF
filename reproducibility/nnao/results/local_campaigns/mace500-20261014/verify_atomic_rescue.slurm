#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao500-atomic-gate
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/mace500-20261014/atomic-gate-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
python -m pytest -q tests/nnao/test_atomic_guess.py tests/nnao/test_fixed_pool_training.py tests/nnao/test_df_rhf_solver.py tests/nnao/test_methane_energy.py
export JAX_PLATFORMS=cuda,cpu
python -u artifacts/mace500-20261014/verify_atomic_rescue.py
