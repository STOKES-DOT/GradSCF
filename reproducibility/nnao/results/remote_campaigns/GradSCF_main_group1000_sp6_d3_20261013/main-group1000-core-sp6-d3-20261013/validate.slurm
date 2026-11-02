#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao1000-check
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/main-group1000-core-sp6-d3-20261013/validate-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=4 JAX_PLATFORMS=cpu
python -m pytest -q tests/nnao/test_cao_rescue.py tests/nnao/test_reference_631g.py tests/nnao/test_basis_family_grid.py tests/nnao/test_expanded_elements.py tests/nnao/test_fixed_pool_training.py tests/nnao/test_pool_workflow.py
export JAX_PLATFORMS=cuda,cpu
for zeta in szp dzp tzp; do
  for tier in s m l; do
    python -u artifacts/main-group1000-core-sp6-d3-20261013/validate_campaign.py nnao_${zeta}_${tier}
  done
done
python -c 'import json; from pathlib import Path; p=Path("artifacts/main-group1000-core-sp6-d3-20261013"); files=list((p/"validation").glob("*/passed.json")); assert len(files)==9; rows=[json.loads(f.read_text()) for f in files]; assert all(r["passed"] for r in rows); (p/"validation-passed.json").write_text(json.dumps({"passed":True,"results":rows},indent=2))'
