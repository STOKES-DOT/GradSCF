#!/bin/bash
#SBATCH --job-name=nnao27-small
#SBATCH --partition=home
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/small-%j.log
set -euo pipefail
benchmark_prefix=/home/user_yuanjiao/private/homefile/conda-envs/gpu4pyscf_bench
source "$benchmark_prefix/etc/conda/activate.d/gpu4pyscf_bench_cuda.sh"
export PATH="$benchmark_prefix/bin:$CUDA_PATH/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
benchmark_cache=/tmp/nnao27-${SLURM_JOB_ID}
mkdir -p "$benchmark_cache/cuda" "$benchmark_cache/cupy"
export CUDA_CACHE_PATH="$benchmark_cache/cuda" CUPY_CACHE_DIR="$benchmark_cache/cupy"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 CUPY_TF32=0
python -u run_gpu.py --lane small --direct
