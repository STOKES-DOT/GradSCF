#!/bin/bash
# Fixed-geometry CPU RI/SCF -> one allocated GPU; separate from CPU timing pairs.
set -u
cd "$HOME/private/homefile/gradscf-gw-family-memory-20261030"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=32 JAX_ENABLE_X64=1 JAX_ENABLE_COMPILATION_CACHE=false
export JAX_PLATFORMS=cuda,cpu PYTHONPATH="$PWD/candidate/src"
PY=../gradscf-libcint-nnao-20261026/env-compare/bin/python
for basis in svp tzvp; do
    repeats=0
    if test "$basis" = svp; then repeats=1; fi
    timeout 6000 "$PY" harness/run.py water "results/phenanthrene-$basis-gpu.jsonl" \
        --case-input "phenanthrene-def2-$basis.json" --nw 64 --threads 32 \
        --device gpu --repeats "$repeats" >"results/phenanthrene-$basis-gpu.log" 2>&1
    code=$?
    echo "$code" >"results/phenanthrene-$basis-gpu.exit"
    if test "$code" -ne 0; then exit "$code"; fi
done
echo 0 >results/gpu-capacity.exit
