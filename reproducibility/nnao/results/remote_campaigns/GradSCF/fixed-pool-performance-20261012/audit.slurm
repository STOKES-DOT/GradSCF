#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-fixed-perf-audit
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/fixed-pool-performance-20261012/audit-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/artifacts/fixed-pool-performance-20261012/source/src:$PWD/artifacts/fixed-pool-performance-20261012/source/src/gradscf/model/nnao/vendor/mace-jax
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export AUDIT_RESULTS=audit-results-mo.json
JAX_PLATFORMS=cpu python -m pytest -q \
  -c artifacts/fixed-pool-performance-20261012/source/pyproject.toml \
  artifacts/fixed-pool-performance-20261012/source/tests/test_df_rhf_stability.py \
  artifacts/fixed-pool-performance-20261012/source/tests/test_df_rhf_stability_curvature.py \
  artifacts/fixed-pool-performance-20261012/source/tests/nnao/test_fixed_pool_training.py \
  artifacts/fixed-pool-performance-20261012/source/tests/nnao/test_native_exponent_scf.py
python -u artifacts/fixed-pool-performance-20261012/audit.py
