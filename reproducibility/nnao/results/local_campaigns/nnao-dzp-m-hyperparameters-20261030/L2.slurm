#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao50-hidden-L2
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/nnao-dzpm-hidden-l50-20261030
#SBATCH --output=logs/L2-%j.log
#SBATCH --open-mode=append
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021/repairs/kr3-20261024/source/src:/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021/repairs/kr3-20261024/source/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/jit-cache/L2
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
date -Is
if [ -f runs/L2/last.npz ]; then
    python -u run_hidden_l.py --profile L2 --resume
else
    python -u run_hidden_l.py --profile L2
fi
date -Is
