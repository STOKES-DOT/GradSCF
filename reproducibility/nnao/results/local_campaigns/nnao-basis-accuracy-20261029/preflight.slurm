#!/bin/bash
#SBATCH --job-name=nnao27-preflight
#SBATCH --partition=home
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/preflight-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 CUPY_TF32=0
python -u run_gpu.py --preflight
