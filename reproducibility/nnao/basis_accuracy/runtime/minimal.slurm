#!/bin/bash
#SBATCH --job-name=gpu4-mace-water-test
#SBATCH --partition=home
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/minimal-%j.log
set -euo pipefail
benchmark_prefix=/home/user_yuanjiao/private/homefile/conda-envs/gpu4pyscf_bench
benchmark_python="$benchmark_prefix/bin/python"
source "$benchmark_prefix/etc/conda/activate.d/gpu4pyscf_bench_cuda.sh"
benchmark_cuda="$CUDA_PATH"
scratch_cache=/tmp/gpu4-water-cache-${SLURM_JOB_ID}
mkdir -p "$scratch_cache/cuda" "$scratch_cache/cupy" "$scratch_cache/jax"
export CUDA_CACHE_PATH="$scratch_cache/cuda" CUPY_CACHE_DIR="$scratch_cache/cupy"
export JAX_COMPILATION_CACHE_DIR="$scratch_cache/jax"
export PATH="$benchmark_cuda/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$benchmark_cuda"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUPY_TF32=0
"$benchmark_python" -u runtime/validate_runtime.py --gpu
status=0
WATER_BASIS_KEY=t14 "$benchmark_python" -u runtime/test_water.py || status=1
WATER_BASIS_KEY=nnao_szp_s "$benchmark_python" -u runtime/test_water.py || status=1
JAX_PLATFORMS=cuda "$benchmark_python" -u runtime/test_mace.py || status=1
exit "$status"
