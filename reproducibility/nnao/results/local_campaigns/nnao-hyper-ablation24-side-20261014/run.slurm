#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-hyper24
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/nnao-hyper-ablation24-side-20261014/campaign-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-hyper-ablation24-side-20261014/jit-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
campaign_status=0
for variant in lr3e4 correlation3 radial64; do
    set +e
    python -u artifacts/nnao-hyper-ablation24-side-20261014/launch.py preflight "$variant" \
        > "artifacts/nnao-hyper-ablation24-side-20261014/${variant}-preflight.log" 2>&1
    preflight_status=$?
    if [ "$preflight_status" -eq 0 ]; then
        python -u artifacts/nnao-hyper-ablation24-side-20261014/launch.py train "$variant" \
            > "artifacts/nnao-hyper-ablation24-side-20261014/${variant}-training.log" 2>&1
        train_status=$?
    else
        train_status=99
    fi
    set -e
    echo "$variant preflight=$preflight_status train=$train_status"
    if [ "$preflight_status" -ne 0 ] || [ "$train_status" -ne 0 ]; then
        campaign_status=1
    fi
done
python artifacts/nnao-hyper-ablation24-side-20261014/summarize.py
exit "$campaign_status"
