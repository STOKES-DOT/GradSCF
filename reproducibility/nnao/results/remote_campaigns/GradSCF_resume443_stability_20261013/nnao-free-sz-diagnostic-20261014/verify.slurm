#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-free-sz-check
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/nnao-free-sz-diagnostic-20261014/verify-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
TASK_CACHE=$(mktemp -d /tmp/nnao-free-sz-check-XXXXXX)
export CUDA_CACHE_PATH=$TASK_CACHE/nvidia CUPY_CACHE_DIR=$TASK_CACHE/cupy
mkdir -p "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR"
python -u artifacts/nnao-free-sz-diagnostic-20261014/verify_frozen.py
