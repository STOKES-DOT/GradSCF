#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-dimer4-gate
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --nodelist=n006
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014
#SBATCH --output=artifacts/nnao-dimer-controls-20261026/smoke-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/tests/comparisons
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
date -Is
python -m pytest -q -o pythonpath= tests/nnao/test_dimer_controls.py
python artifacts/nnao-dimer-controls-20261026/measure.py \
  --output artifacts/nnao-dimer-controls-20261026/run/resource-${SLURM_JOB_ID}.json \
  -- python -u tests/comparisons/nnao_dimer_controls/run.py \
  --inputs artifacts/nnao-dimer-controls-20261026/inputs/cases.json \
  --output artifacts/nnao-dimer-controls-20261026/run --max-cases 4
python -c 'import json; x=json.load(open("artifacts/nnao-dimer-controls-20261026/run/gate.json")); assert x["status"]=="passed"'
date -Is
