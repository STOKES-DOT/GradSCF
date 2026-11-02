#!/bin/bash
#SBATCH --job-name=nnao27-qz
#SBATCH --partition=home
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=logs/qz-%j.log
set -euo pipefail
benchmark_prefix=/home/user_yuanjiao/private/homefile/conda-envs/gpu4pyscf_bench
source "$benchmark_prefix/etc/conda/activate.d/gpu4pyscf_bench_cuda.sh"
export PATH="$benchmark_prefix/bin:$CUDA_PATH/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
benchmark_cache=/tmp/nnao27-${SLURM_JOB_ID}
mkdir -p "$benchmark_cache/cuda" "$benchmark_cache/cupy"
export CUDA_CACHE_PATH="$benchmark_cache/cuda" CUPY_CACHE_DIR="$benchmark_cache/cupy"
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 CUPY_TF32=0
python -u run_gpu.py --lane qz --direct
