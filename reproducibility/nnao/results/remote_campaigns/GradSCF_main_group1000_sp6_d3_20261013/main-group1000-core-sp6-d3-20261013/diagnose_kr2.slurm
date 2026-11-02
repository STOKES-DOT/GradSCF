#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-kr2-audit
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/main-group1000-core-sp6-d3-20261013/kr2-audit-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
python -u artifacts/main-group1000-core-sp6-d3-20261013/diagnose_kr2.py
