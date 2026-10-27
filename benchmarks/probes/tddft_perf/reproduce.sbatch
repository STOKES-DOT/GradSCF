#!/bin/bash
# Submit from a private/homefile source checkout after reading the live manual.
# Set PROBE_PYTHON to the existing comparison Python environment.
#SBATCH --partition=home
#SBATCH --nodelist=n005
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --job-name=tddft-probe-reproduce
#SBATCH --output=tddft-probe-reproduce-%j.log
set -euo pipefail
: "${PROBE_PYTHON:?Set PROBE_PYTHON to the existing environment Python}"
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
result_root=reports/tddft_perf/20261027-33a7cab
output="reports/tddft_perf/rerun-${SLURM_JOB_ID}"
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
    "$PROBE_PYTHON" benchmarks/probes/tddft_perf/run.py \
    "$result_root/matrix-reproduce.json" "$output" --seconds 7200
"$PROBE_PYTHON" benchmarks/probes/tddft_perf/analyze.py "$output"
