#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=ethane-STO6G-SZP
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/nnao-ethane-sto6g-polarized-20261020/run-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u artifacts/nnao-ethane-sto6g-polarized-20261020/run.py
