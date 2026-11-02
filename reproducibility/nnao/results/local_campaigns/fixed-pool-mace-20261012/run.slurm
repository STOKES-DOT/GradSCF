#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-fixed-pool-gate
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/fixed-pool-mace-20261012/slurm-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m gradscf.integrals._native.build --jobs 8 --build-dir artifacts/fixed-pool-mace-20261012/native-build --blas-library /usr/lib64/libblas.so
JAX_PLATFORMS=cpu python -m pytest -q tests/nnao/test_fixed_pool_training.py tests/nnao/test_primitive_pool.py tests/nnao/test_pool_objective.py tests/nnao/test_shared_pilot_training.py
python -m gradscf.model.nnao.workflows.fixed_pool --platform cuda --candidate artifacts/fixed-pool-mace-20261012/candidate-exponents.json --cache-dir artifacts/fixed-pool-mace-20261012/cache --output-dir artifacts/fixed-pool-mace-20261012/train --epochs 5
python -m gradscf.model.nnao.workflows.fixed_pool --platform cuda --candidate artifacts/fixed-pool-mace-20261012/candidate-exponents.json --cache-dir artifacts/fixed-pool-mace-20261012/cache --output-dir artifacts/fixed-pool-mace-20261012/train --epochs 6 --resume
