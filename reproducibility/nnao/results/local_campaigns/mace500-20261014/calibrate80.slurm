#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao500-pool-L80
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/mace500-20261014/calibration80-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
python -m pytest -q tests/nnao/test_minibatch_training.py tests/nnao/test_mace_explicit_architecture.py
export JAX_PLATFORMS=cuda,cpu
python -u artifacts/mace500-20261014/probe_pool_memory.py
python -u -m gradscf.model.nnao.workflows.pool_calibration \
  --references artifacts/mace500-20261014/inputs/pilot.json \
  --output-dir artifacts/mace500-20261014/calibration \
  --calibration-steps 20 --contraction-steps 300 --starts 3 --seed 0 --scf-platform cuda
export JAX_PLATFORMS=cpu
python -m gradscf.model.nnao.workflows.basis_grid select \
  --pilot-dir artifacts/mace500-20261014/calibration \
  --references artifacts/mace500-20261014/inputs/pilot.json \
  --output-dir artifacts/mace500-20261014/selection
