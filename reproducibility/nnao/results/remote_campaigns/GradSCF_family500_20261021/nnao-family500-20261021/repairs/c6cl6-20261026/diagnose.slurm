#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-c6cl6-metric
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --nodelist=n006
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021
#SBATCH --output=artifacts/nnao-family500-20261021/repairs/c6cl6-20261026/diagnose-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
source_root=$PWD/artifacts/nnao-family500-20261021/repairs/ne6-20261023/source
export PYTHONPATH=$source_root/src:$source_root/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
date -Is
python -u artifacts/nnao-family500-20261021/repairs/c6cl6-20261026/diagnose.py
date -Is
