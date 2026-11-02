#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-bench200-e594
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/nnao-benchmark200-20261014/benchmark-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
TASK_CACHE=$(mktemp -d /tmp/nnao-bench200-594-XXXXXX)
export CUDA_CACHE_PATH=$TASK_CACHE/nvidia CUPY_CACHE_DIR=$TASK_CACHE/cupy
mkdir -p "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR"
python -u artifacts/nnao-benchmark200-20261014/native_forward.py
python -u artifacts/nnao-benchmark200-20261014/gpu4_reference.py \
  --structures artifacts/nnao-benchmark200-20261014/test-selection.jsonl \
  --frozen-bases artifacts/nnao-benchmark200-20261014/frozen-bases.jsonl \
  --output artifacts/nnao-benchmark200-20261014/gpu4 \
  --bases minao sto-3g 3-21g 6-31g '6-31g(d)' '6-31g(d,p)' def2-svp cc-pvdz \
    '6-311g(d,p)' def2-tzvp def2-tzvpp cc-pvtz NNAO-pilot-best975 NNAO-200-initial NNAO-200-epoch594
