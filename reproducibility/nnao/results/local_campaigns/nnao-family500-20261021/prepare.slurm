#!/bin/bash
#SBATCH --partition=home
#SBATCH --job-name=prepare
#SBATCH --gres=gpu:A800:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=4:00:00
#SBATCH --chdir=/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021
#SBATCH --output=artifacts/nnao-family500-20261021/prepare-%j.log
#SBATCH --nodelist=n005
set -euo pipefail
export PATH=/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=4
export JAX_COMPILATION_CACHE_DIR=$PWD/artifacts/nnao-family500-20261021/jit-cache/prepare
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
date -Is
export CC=/usr/bin/gcc CXX=/usr/bin/g++
"$CXX" --version
JAX_PLATFORMS=cpu python -m gradscf.integrals._native.build --jobs 8 --build-dir artifacts/nnao-family500-20261021/native-build
JAX_PLATFORMS=cpu python -m pytest -q tests/nnao/test_energy_stopping.py tests/nnao/test_minibatch_training.py tests/nnao/test_single_pass_training.py tests/nnao/test_fixed_pool_training.py tests/nnao/test_pool_selection_checks.py -k 'not real_mace'
python -u artifacts/nnao-family500-20261021/campaign.py prepare
date -Is
