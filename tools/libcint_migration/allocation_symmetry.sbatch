#!/bin/bash
#SBATCH --partition=home
#SBATCH --nodelist=n006
#SBATCH --gres=gpu:NV5090:1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=libcint-ad-symmetry
#SBATCH --output=allocation-symmetry-%j.log
set -euo pipefail
printf 'job=%s node=%s gpu=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"
touch "allocation-${SLURM_JOB_ID}.ready"
while [[ ! -f "allocation-${SLURM_JOB_ID}.stop" ]]; do sleep 5; done
