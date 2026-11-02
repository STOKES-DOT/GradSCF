#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=nnao-hyper24-to1000
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/nnao-hyper1000-side-20261014/campaign-%j.log
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
CONTINUATION_ROOT=/home/user_yuanjiao/private/homefile/GradSCF_mace_ablation24_side_20261014/artifacts/nnao-hyper1000-side-20261014
campaign_status=0
for variant in lr3e4 radial64 correlation3 baseline; do
    if [ "$variant" = baseline ]; then
        checkout=/home/user_yuanjiao/private/homefile/GradSCF_resume443_stability_20261013
    else
        checkout=/home/user_yuanjiao/private/homefile/GradSCF_mace_ablation24_side_20261014
    fi
    export PYTHONPATH=$checkout/src:$checkout/src/gradscf/model/nnao/vendor/mace-jax
    export JAX_COMPILATION_CACHE_DIR=$CONTINUATION_ROOT/jit-cache/$variant
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
    set +e
    python -u "$CONTINUATION_ROOT/resume.py" "$variant" > "$CONTINUATION_ROOT/$variant.log" 2>&1
    result=$?
    set -e
    echo "$variant exit=$result"
    if [ "$result" -ne 0 ]; then campaign_status=1; fi
done
python "$CONTINUATION_ROOT/summarize.py"
exit "$campaign_status"
