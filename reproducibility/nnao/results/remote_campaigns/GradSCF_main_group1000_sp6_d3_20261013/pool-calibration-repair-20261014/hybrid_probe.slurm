#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-pool-gpu-audit
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/pool-calibration-repair-20261014/gpu-audit-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
python -u artifacts/pool-calibration-repair-20261014/monitor.py --gpu
