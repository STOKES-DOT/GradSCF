#!/bin/bash
# Execute from the campaign root in one existing allocation, before other work.
set -eu
allocation=$1
python=../gradscf-libcint-nnao-20261026/env-compare/bin/python
runner=harness/stationarity-v9.py
printf '%s  %s\n' 5b5abca5a79d9362515820b9530d64b5790504cb9cbf3c98ae520bee47534853 \
    "$runner" | sha256sum -c -
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=32
for variant in candidate-brillouin candidate-grid; do
    export PYTHONPATH="$variant/src"
    output="results/phi-fock-$allocation-$variant"
    test ! -e "$output.process.json"
    srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c64 \
        "$python" harness/bounded.py --seconds 6000 "$output.process.json" -- \
        "$python" "$runner" results/phi-tzvp-pbe0.npz phenanthrene-def2-tzvp.json \
        results/phi-tzvp-pbe0-g0w0-gpu.jsonl "$output.json" --threads 32 --fock-repeats 3 \
        > "$output.log" 2>&1
done
srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c1 \
    "$python" harness/compare_fock.py \
    "results/phi-fock-$allocation-candidate-brillouin.json" \
    "results/phi-fock-$allocation-candidate-grid.json" harness/fock_acceptance.json \
    "results/phi-fock-$allocation-comparison.json"
