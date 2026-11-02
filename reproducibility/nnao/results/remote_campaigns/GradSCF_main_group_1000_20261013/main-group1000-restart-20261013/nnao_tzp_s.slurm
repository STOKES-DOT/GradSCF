#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao1000-tzp-s
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/main-group1000-restart-20261013/nnao_tzp_s-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/main-group1000-restart-20261013/jit-cache/nnao_tzp_s
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -c 'import json; from pathlib import Path; from gradscf.model.nnao.runtime import scientific_source_hash, repository_root; p=Path("artifacts/main-group1000-restart-20261013"); d=json.loads((p/"validation-passed.json").read_text()); h=scientific_source_hash(repository_root()); assert d["passed"] and len(d["results"])==9 and all(r["scientific_source_sha256"]==h for r in d["results"]), "Validation/source mismatch"'
python -u -m gradscf.model.nnao.workflows.basis_grid run --family nnao_tzp_s --references artifacts/main-group1000-restart-20261013/inputs/references.json --pilot-references artifacts/main-group1000-restart-20261013/inputs/pilot.json --output-dir artifacts/main-group1000-restart-20261013/runs/nnao_tzp_s --epochs 1000 --calibration-steps 20 --contraction-steps 300 --starts 3
