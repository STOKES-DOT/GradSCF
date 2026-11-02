#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-free-dz-7
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/nnao-free-dz-diagnostic-20261014/diagnostic-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_PLATFORMS=cpu
python -m pytest -q artifacts/nnao-free-dz-diagnostic-20261014/test_expand.py
export JAX_PLATFORMS=cuda,cpu
python -u artifacts/nnao-free-dz-diagnostic-20261014/run_diagnostic.py
TASK_CACHE=$(mktemp -d /tmp/nnao-free-dz-check-XXXXXX)
export CUDA_CACHE_PATH=$TASK_CACHE/nvidia CUPY_CACHE_DIR=$TASK_CACHE/cupy
mkdir -p "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR"
set +e
python -u artifacts/nnao-free-dz-diagnostic-20261014/verify_frozen.py
verification_status=$?
set -e
python artifacts/nnao-free-dz-diagnostic-20261014/summarize.py
exit "$verification_status"
