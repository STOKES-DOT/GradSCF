#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao500-relative-ref
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/nnao-relative500-20261018/reference-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u artifacts/nnao-relative500-20261018/gpu4_check.py
python -u artifacts/nnao-relative500-20261018/summarize.py
