#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-mace-capacity24
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/nnao-mace-capacity24-side-20261014/campaign-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-mace-capacity24-side-20261014/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -u artifacts/nnao-mace-capacity24-side-20261014/launch.py preflight
set +e
python -u artifacts/nnao-mace-capacity24-side-20261014/launch.py default \
    > artifacts/nnao-mace-capacity24-side-20261014/default.log 2>&1
default_status=$?
python -u artifacts/nnao-mace-capacity24-side-20261014/launch.py small \
    > artifacts/nnao-mace-capacity24-side-20261014/small.log 2>&1
small_status=$?
set -e
python artifacts/nnao-mace-capacity24-side-20261014/summarize.py
test "$default_status" -eq 0 && test "$small_status" -eq 0
