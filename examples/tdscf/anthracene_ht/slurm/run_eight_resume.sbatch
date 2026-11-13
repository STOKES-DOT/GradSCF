#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=anthracene-eight-HT-resume
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:NV5090:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
RUN=/home/user_yuanjiao/private/homefile/gradscf-anthracene-ht-20261110
PY=/home/user_yuanjiao/private/homefile/gradscf-libcint-nnao-20261026/env-compare/bin/python
cd "$RUN"
export PYTHONPATH="$RUN/src:$RUN:$RUN/tests" JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1
export JAX_ENABLE_COMPILATION_CACHE=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=64 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export GRADSCF_ANTHRACENE_INPUT="$RUN/examples/tdscf/anthracene_ht/input_eight_states.json"
export GRADSCF_ANTHRACENE_OUTPUT="$RUN/reports/anthracene_ht/20261110-b3lyp-def2svp/eight_states/raw"
scontrol show job "$SLURM_JOB_ID" > "results/eight-resume-allocation-$SLURM_JOB_ID.txt"
# Only completed, signed point files are reused; the source/geometry/native
# identity checks in the unchanged driver remain authoritative.
test ! -e "$GRADSCF_ANTHRACENE_OUTPUT/mode_response_failure.json"
srun -n1 -c64 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard --seconds 82800 --rss-bytes 236223201280 results/eight-mode-resume.process.json -- "$PY" examples/tdscf/anthracene_ht/03_mode_response.py > results/eight-mode-resume.log 2>&1
export JAX_PLATFORMS=cpu
srun -n1 -c8 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard --seconds 300 --rss-bytes 21474836480 results/eight-spectrum-resume.process.json -- "$PY" examples/tdscf/anthracene_ht/04_spectrum.py > results/eight-spectrum-resume.log 2>&1
