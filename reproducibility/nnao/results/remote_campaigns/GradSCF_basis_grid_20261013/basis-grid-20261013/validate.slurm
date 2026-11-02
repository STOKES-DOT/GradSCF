#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-grid-check
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/basis-grid-20261013/validate-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=4
export JAX_PLATFORMS=cpu
python -m pytest -q tests/nnao/test_basis_family_grid.py tests/nnao/test_fixed_pool_training.py tests/nnao/test_primitive_pool.py tests/nnao/test_pool_workflow.py
export JAX_PLATFORMS=cuda,cpu
for zeta in szp dzp tzp; do
  for tier in s m l; do
    python -u artifacts/basis-grid-20261013/smoke_grid.py nnao_${zeta}_${tier}
  done
done
python -c 'from pathlib import Path; import json; p=Path("artifacts/basis-grid-20261013/smoke"); f=list(p.glob("*/training/gate.json")); assert len(f)==9 and all(json.loads(x.read_text())["passed"] for x in f); Path("artifacts/basis-grid-20261013/validation-passed.json").write_text(json.dumps({"passed":True,"families":sorted(x.parent.parent.name for x in f)}))'
