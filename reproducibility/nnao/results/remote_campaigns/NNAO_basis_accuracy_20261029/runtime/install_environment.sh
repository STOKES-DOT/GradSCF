#!/bin/bash
set -euo pipefail
module load anaconda3
benchmark_root=/home/user_yuanjiao/private/homefile/NNAO_basis_accuracy_20261029
benchmark_prefix=/home/user_yuanjiao/private/homefile/conda-envs/gpu4pyscf_bench
benchmark_toolkit=/opt/data/hpc/common/softwares/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
test -x "$benchmark_toolkit/bin/nvcc"
"$benchmark_toolkit/bin/nvcc" --version
export PIP_CACHE_DIR="$benchmark_root/runtime/pip-cache"
export CONDA_PKGS_DIRS="$benchmark_root/runtime/conda-pkgs"
export CONDA_NO_LOCK=true
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$benchmark_root"
export PWD="$benchmark_root"
date -Is
if [[ ! -x "$benchmark_prefix/bin/python" ]]; then
  conda create --prefix "$benchmark_prefix" --yes --quiet --repodata-fn=current_repodata.json python=3.11 pip \
    --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
fi
"$benchmark_prefix/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1'
shopt -s nullglob
cached_wheels=("$benchmark_root"/runtime/prefetched-wheels/jax_cuda12_pjrt-0.10.0-*.whl)
"$benchmark_prefix/bin/python" -m pip install --index-url https://mirrors.aliyun.com/pypi/simple/ \
    -r runtime/requirements-official-v1.7.0.txt -r runtime/requirements-cuda124.txt \
    -r runtime/requirements-mace-jax.txt "${cached_wheels[@]}" \
    --find-links runtime/prefetched-wheels --report runtime/install-report.json
"$benchmark_prefix/bin/python" -m pip check
"$benchmark_prefix/bin/python" runtime/register_source.py
source "$benchmark_prefix/etc/conda/activate.d/gpu4pyscf_bench_cuda.sh"
CUDA_VISIBLE_DEVICES='' "$benchmark_prefix/bin/python" runtime/validate_runtime.py
"$benchmark_prefix/bin/python" -m pip freeze > runtime/environment-freeze.txt
date -Is
echo ENVIRONMENT_INSTALL_PASSED
