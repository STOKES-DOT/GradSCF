#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao_tzp_m
#SBATCH --gres=gpu:A100_80G:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=artifacts/basis-grid-20261013/nnao_tzp_m-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/basis-grid-20261013/jit-cache/nnao_tzp_m
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -c 'import json; assert json.load(open("artifacts/basis-grid-20261013/validation-passed.json"))["passed"]'
python -c 'import json; from pathlib import Path; from gradscf.model.nnao.runtime import scientific_source_hash, repository_root; h=scientific_source_hash(repository_root()); files=list(Path("artifacts/basis-grid-20261013/smoke").glob("*/training/metadata.json")); assert len(files)==9 and all(json.loads(p.read_text())["identity"]["scientific_source_sha256"]==h for p in files), "Validation source differs from training source"'
python -u -m gradscf.model.nnao.workflows.basis_grid run --family nnao_tzp_m --references /home/user_yuanjiao/private/homefile/GradSCF/artifacts/fixed-pool200-20261012/references.json --output-dir artifacts/basis-grid-20261013/runs/nnao_tzp_m --epochs 1000 --calibration-steps 20 --contraction-steps 300 --starts 3
