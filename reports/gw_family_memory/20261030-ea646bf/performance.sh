#!/bin/bash
# Run once, INSIDE the granted allocation. Numerical jobs are strictly serial.
set -u
cd "$HOME/private/homefile/gradscf-gw-family-memory-20261030"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=32 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export JAX_ENABLE_COMPILATION_CACHE=false
PY=../gradscf-libcint-nnao-20261026/env-compare/bin/python
while [ ! -f candidate-ready ]; do sleep 2; done
run() {
    label=$1
    shift
    timeout 6000 "$PY" "$@" > "results/$label.log" 2>&1
    code=$?
    echo "$code" > "results/$label.exit"
}
for variant in baseline candidate; do
    export PYTHONPATH="$PWD/$variant/src"
    for kind in qsgw scgw; do
        run "$variant-$kind-cpu1" harness/matrix.py "$kind" "results/$variant-$kind-cpu1.jsonl" --threads 1
    done
    run "$variant-charge-cpu1" harness/matrix.py charge "results/$variant-charge-cpu1.jsonl" --threads 1 --nmo 2
    for method in evgw0 evgw; do
        run "$variant-ch2o-$method-cpu32" harness/run.py water "results/$variant-ch2o-$method-cpu32.jsonl" \
            --method "$method" --case-input ch2o-def2-svp.json --nw 32 --threads 32 --repeats 3 --device cpu
    done
done
export PYTHONPATH="$PWD/candidate/src"
run candidate-phenanthrene-g0w0 harness/run.py water results/candidate-phenanthrene-g0w0.jsonl \
    --case-input phenanthrene-def2-svp.json --nw 64 --threads 32 --repeats 1 --device cpu
run candidate-water-conventional harness/run.py water results/candidate-water-conventional.jsonl \
    --case-input water-def2-svp.json --conventional packed --nw 64 --threads 32 --repeats 3 --device cpu
echo done > results/performance.done
