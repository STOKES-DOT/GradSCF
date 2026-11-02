#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-curves-barriers
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/nnao-curves-barriers-20261020/representative-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax:$PWD/tests/comparisons
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-curves-barriers-20261020/jit-cache
python -c 'import json; d=json.load(open("artifacts/nnao-curves-barriers-20261020/smoke/results.json")); assert d["status"] == "complete" and d["passed"] == 39 and d["failed"] == 0, "Short numerical validation did not pass all 39 cases"'
python -u tests/comparisons/nnao_reactivity/run.py \
  --dataset artifacts/nnao-curves-barriers-20261020/dataset.json \
  --output artifacts/nnao-curves-barriers-20261020/representative \
  --checkpoint artifacts/nnao-relative500-20261018/checkpoint-current.npz \
  --snapshot artifacts/nnao-relative500-20261018/snapshot.json \
  --profile representative
