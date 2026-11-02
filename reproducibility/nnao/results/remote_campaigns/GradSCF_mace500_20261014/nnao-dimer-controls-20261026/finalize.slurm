#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-dimer4-figures
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nodelist=n006
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014
#SBATCH --output=artifacts/nnao-dimer-controls-20261026/figures-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/tests/comparisons
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is
python artifacts/nnao-dimer-controls-20261026/assemble.py
python artifacts/nnao-dimer-controls-20261026/plot_controls.py
python artifacts/nnao-dimer-controls-20261026/verify_results.py
date -Is
