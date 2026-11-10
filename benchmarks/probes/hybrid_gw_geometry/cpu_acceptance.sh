#!/bin/bash
# Run serially inside the campaign's existing single-GPU/64-CPU allocation.
set -eu
allocation=$1
python=../gradscf-libcint-nnao-20261026/env-compare/bin/python
export JAX_ENABLE_X64=1 JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=32 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu PYTHONPATH=candidate-quota-v1/src
output=results/phi-fock-quota-$allocation
test ! -e "$output.process.json"
srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c64 \
    "$python" harness/bounded.py --seconds 6000 "$output.process.json" -- \
    "$python" harness/stationarity-v9.py results/phi-tzvp-pbe0.npz \
    phenanthrene-def2-tzvp.json results/phi-tzvp-pbe0-g0w0-gpu.jsonl \
    "$output.json" --threads 32 --fock-repeats 3 > "$output.log" 2>&1

# Test all supported UKS XC families with the integrated production library.
output=results/uks-four-xc-$allocation
srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c8 \
    "$python" harness/bounded.py --seconds 1800 "$output.process.json" -- \
    "$python" -m pytest -q -c candidate-quota-v1/pyproject.toml \
    candidate-quota-v1/tests/test_molecular_uks_scf_forward.py \
    --junitxml="$output.xml" > "$output.log" 2>&1

# Each pair has identical synthetic inputs and a fresh process. These are
# conventional packed-array kernels, not end-to-end molecular calculations.
for specification in 'gks-pure 30 cpu ad' 'gks-hybrid 90 cpu value' \
                     'gks-energy 30 gpu ad' 'uks-jk 30 cpu ad'; do
    read -r kernel nao device mode <<< "$specification"
    for variant in candidate-grid candidate-quota-v1; do
        export PYTHONPATH="$variant/src"
        gres=none
        options=()
        if test "$device" = gpu; then
            export JAX_PLATFORMS=cuda,cpu
            gres=gpu:NV5090:1
        else
            export JAX_PLATFORMS=cpu
        fi
        if test "$mode" = ad; then options+=(--ad); fi
        output="results/spin-$kernel-n$nao-$device-$variant-$allocation"
        test ! -e "$output.jsonl"
        srun --jobid="$allocation" --exact --gres="$gres" -N1 -n1 -c64 \
            "$python" harness/bounded.py --seconds 600 "$output.process.json" -- \
            "$python" harness/spin-kernels-v1.py "$output.jsonl" \
            --kernel "$kernel" --nao "$nao" --device "$device" \
            --threads 8 --repeats 10 "${options[@]}" > "$output.log" 2>&1
    done
done
