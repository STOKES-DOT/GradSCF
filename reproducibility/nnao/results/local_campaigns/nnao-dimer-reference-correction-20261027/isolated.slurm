#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-isolated-refs
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --nodelist=n006
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014
#SBATCH --output=artifacts/nnao-dimer-reference-correction-20261027/isolated-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/tests/comparisons
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
date -Is
python -u artifacts/nnao-dimer-reference-correction-20261027/isolated.py
date -Is
