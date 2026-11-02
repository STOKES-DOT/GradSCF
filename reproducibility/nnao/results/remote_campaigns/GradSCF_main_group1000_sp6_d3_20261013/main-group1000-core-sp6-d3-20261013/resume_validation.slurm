#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-resume-check
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/main-group1000-core-sp6-d3-20261013/resume-validation-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_PLATFORMS=cpu
python -m pytest -q tests/solvers/test_spectral_density.py tests/solvers/test_gmres_accuracy.py tests/nnao/test_ge2h6_implicit.py tests/nnao/test_kr2_stability.py tests/nnao/test_cao_rescue.py tests/test_df_rhf_stability.py tests/nnao/test_fixed_pool_training.py tests/nnao/test_basis_family_grid.py
export JAX_PLATFORMS=cuda,cpu
python -m pytest -q tests/solvers/test_spectral_density.py tests/nnao/test_ge2h6_implicit.py tests/nnao/test_kr2_stability.py
for family in nnao_dzp_s nnao_szp_s nnao_szp_m nnao_szp_l nnao_dzp_m nnao_dzp_l nnao_tzp_s nnao_tzp_m nnao_tzp_l; do
  python -u artifacts/main-group1000-core-sp6-d3-20261013/resume_validation.py "$family"
done
python -c 'import json; from pathlib import Path; from gradscf.model.nnao.runtime import scientific_source_hash,repository_root; p=Path("artifacts/main-group1000-core-sp6-d3-20261013"); rows=[json.loads(f.read_text()) for f in (p/"validation-ge2h6-fixed").glob("*/passed.json")]; h=scientific_source_hash(repository_root()); assert len(rows)==9 and all(r["passed"] and r["scientific_source_sha256"]==h for r in rows); (p/"validation-passed.json").write_text(json.dumps({"passed":True,"results":rows},indent=2))'
