#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-relative200-e629
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/nnao-relative200-20261014/benchmark-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
TASK_CACHE=$(mktemp -d /tmp/nnao-relative200-XXXXXX)
export CUDA_CACHE_PATH=$TASK_CACHE/nvidia CUPY_CACHE_DIR=$TASK_CACHE/cupy
mkdir -p "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR"
python -u artifacts/nnao-relative200-20261014/native_forward.py
set +e
python -u artifacts/nnao-relative200-20261014/gpu4_reference.py \
  --structures artifacts/nnao-relative200-20261014/test-selection.jsonl \
  --frozen-bases artifacts/nnao-relative200-20261014/frozen-bases.jsonl \
  --output artifacts/nnao-relative200-20261014/gpu4 \
  --bases minao 3-21g '6-31g(d,p)' def2-svp cc-pvdz cc-pvtz NNAO-200-initial NNAO-200-best
reference_status=$?
set -e
python artifacts/nnao-relative200-20261014/summarize.py
exit "$reference_status"
