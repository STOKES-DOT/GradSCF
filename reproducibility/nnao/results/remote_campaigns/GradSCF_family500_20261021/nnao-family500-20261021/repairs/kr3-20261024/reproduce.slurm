#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-kr3-reproduce
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021
#SBATCH --output=artifacts/nnao-family500-20261021/repairs/kr3-20261024/reproduce-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/artifacts/nnao-family500-20261021/repairs/ne6-20261023/source/src:$PWD/artifacts/nnao-family500-20261021/repairs/ne6-20261023/source/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
date -Is
python -u artifacts/nnao-family500-20261021/repairs/kr3-20261024/reproduce.py
date -Is
