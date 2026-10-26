#!/bin/bash
#SBATCH --partition=home
#SBATCH --nodelist=n005
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --job-name=gradscf-libcint
#SBATCH --output=allocation-%j.log
set -euo pipefail
printf 'job=%s node=%s gpu=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$CUDA_VISIBLE_DEVICES"
touch allocation.ready
while [[ ! -f allocation.stop ]]; do sleep 5; done
