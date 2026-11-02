#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-hcnts-energy-check
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/nnao-free500-hcnts-20261018/check-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u artifacts/nnao-free500-hcnts-20261018/gpu4_check.py
