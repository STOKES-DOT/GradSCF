#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-reactivity-plots
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=artifacts/nnao-curves-barriers-20261020/plots-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tests/comparisons/nnao_reactivity/plot.py \
  artifacts/nnao-curves-barriers-20261020/representative \
  --dataset artifacts/nnao-curves-barriers-20261020/dataset.json
