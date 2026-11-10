#!/bin/bash
# One successor allocation for the user-requested UKS/GKS and CPU phase.
#SBATCH --partition=home
#SBATCH --job-name=spin-cpu-optimization
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:NV5090:1
#SBATCH --time=04:00:00
#SBATCH --output=spin-cpu-%j.log
set -eu
cd "$SLURM_SUBMIT_DIR"
hostname
printf '%s\n' "$SLURM_JOB_ID" > allocation-spin.id
lscpu > hardware-spin-cpu.txt
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
while test ! -f release.spin; do sleep 30; done
