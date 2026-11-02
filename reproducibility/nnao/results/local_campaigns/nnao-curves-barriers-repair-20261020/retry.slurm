#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-native-scf-repair
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/nnao-curves-barriers-repair-20261020/retry-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax:$PWD/tests/comparisons
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-curves-barriers-repair-20261020/jit-cache
JAX_PLATFORMS=cpu python -m pytest -q tests/nnao/test_reactivity_rescue.py tests/nnao/test_reactivity_merge.py
python -u tests/comparisons/nnao_reactivity/retry.py \
  --parent artifacts/nnao-curves-barriers-20261020/representative \
  --dataset artifacts/nnao-curves-barriers-20261020/dataset.json \
  --selection artifacts/nnao-curves-barriers-repair-20261020/selection.json \
  --output artifacts/nnao-curves-barriers-repair-20261020/retry \
  --checkpoint artifacts/nnao-relative500-20261018/checkpoint-current.npz \
  --snapshot artifacts/nnao-relative500-20261018/snapshot.json
python tests/comparisons/nnao_reactivity/merge.py \
  --parent artifacts/nnao-curves-barriers-20261020/representative \
  --retry artifacts/nnao-curves-barriers-repair-20261020/retry \
  --dataset artifacts/nnao-curves-barriers-20261020/dataset.json \
  --output artifacts/nnao-curves-barriers-repair-20261020/merged
python tests/comparisons/nnao_reactivity/plot.py \
  artifacts/nnao-curves-barriers-repair-20261020/merged \
  --dataset artifacts/nnao-curves-barriers-20261020/dataset.json
python -u artifacts/nnao-curves-barriers-repair-20261020/check_repaired.py \
  --parent artifacts/nnao-curves-barriers-20261020/representative \
  --retry artifacts/nnao-curves-barriers-repair-20261020/retry \
  --selection artifacts/nnao-curves-barriers-repair-20261020/selection.json
