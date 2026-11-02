#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-tzp-cgraph-off
#SBATCH --gres=gpu:A800:1
#SBATCH --nodelist=n005
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=8:00:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021
#SBATCH --output=artifacts/nnao-family500-20261021/diagnostics/cuda-graph-off-20261022/control-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-family500-20261021/jit-cache/cuda-graph-off-s-tzp
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
date -Is
python -u artifacts/nnao-family500-20261021/diagnostics/cuda-graph-off-20261022/run.py
date -Is
