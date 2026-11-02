#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao_tzp_l_metricfix
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --exclude=n005
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021
#SBATCH --output=artifacts/nnao-family500-20261021/repairs/c6cl6-20261026/nnao_tzp_l-%j.log
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
source_root=$PWD/artifacts/nnao-family500-20261021/repairs/c6cl6-20261026/source
export PYTHONPATH=$source_root/src:$source_root/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-family500-20261021/jit-cache/c6cl6fix/nnao_tzp_l
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
date -Is
requeue_for_time_limit() {
    trap - USR1
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue_for_time_limit USR1
python -u artifacts/nnao-family500-20261021/repairs/c6cl6-20261026/train.py &
worker=$!
wait "$worker"
trap - USR1
date -Is
