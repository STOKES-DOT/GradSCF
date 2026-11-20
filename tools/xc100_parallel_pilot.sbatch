#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=xc100-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00

set -euo pipefail
root=${XC100_RUN_ROOT:-/home/user_yuanjiao/private/homefile/gradscf-iks-independent-20261118-r2}
python=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin/python
cd "$root"
export PYTHONPATH="$root/source/src" JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
srun --cpu-bind=threads "$python" -u source/tools/xc100_parallel_worker.py "$root"
