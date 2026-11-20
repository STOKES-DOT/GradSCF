#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=gradscf-iks-open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/user_yuanjiao/private/homefile/gradscf-iks-open-20261117-r2/logs/slurm-%j.out
#SBATCH --error=/home/user_yuanjiao/private/homefile/gradscf-iks-open-20261117-r2/logs/slurm-%j.err

set -euo pipefail
ROOT=/home/user_yuanjiao/private/homefile/gradscf-iks-open-20261117-r2
PYTHON=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/source/src"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
printf '{"job_id":"%s","node":"%s","cpus":%s,"memory":"24G","gpu":"V100:1"}\n' \
    "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$SLURM_CPUS_PER_TASK" > allocation.json
printf '{"phase":"validating","job_id":"%s","completed_baselines":0}\n' "$SLURM_JOB_ID" > status.json

# Unit/acceptance steps run serially inside this same allocation. Model calls
# stay local. Source readiness is published only after validation succeeds.
for attempt in $(seq 1 1440); do
    if test -f release.flag; then exit 0; fi
    if test -f start-worker.flag; then
        srun --cpu-bind=threads "$PYTHON" -u source/tools/xc100_worker.py "$ROOT"
        exit 0
    fi
    sleep 5
done
echo 'No validated worker within two hours; releasing allocation.' >&2
exit 2
