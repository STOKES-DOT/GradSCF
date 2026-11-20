#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=gradscf-xc-gridbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user_yuanjiao/private/homefile/gradscf-xc100-gridbench-20261117/logs/slurm-%j.out
#SBATCH --error=/home/user_yuanjiao/private/homefile/gradscf-xc100-gridbench-20261117/logs/slurm-%j.err

set -euo pipefail
cd /home/user_yuanjiao/private/homefile/gradscf-xc100-gridbench-20261117
printf '{"job_id":"%s","node":"%s","cpus":%s,"memory":"24G","gpu":"V100:1"}\n' \
    "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$SLURM_CPUS_PER_TASK" > allocation.json

# The local controller dispatches one test/benchmark step at a time inside
# this allocation. Completion or an unrecoverable error releases it via flag.
for attempt in $(seq 1 2880); do
    if test -f release.flag; then exit 0; fi
    sleep 5
done
echo 'Benchmark controller lease expired; releasing allocation.' >&2
exit 2
