#!/bin/bash
# Continue only after the serial spin acceptance controller has exited.
set -eu
allocation=$1
stage=${2:-vjp}
python=../gradscf-libcint-nnao-20261026/env-compare/bin/python
export JAX_ENABLE_X64=1 JAX_ENABLE_COMPILATION_CACHE=false JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=32
if test "$stage" = parameters; then
    root=$PWD
    python=$(realpath "$python")
    output="$root/results/parameter-guards-$allocation"
    test ! -e "$output.xml"
    srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c8 \
        --chdir="$root/candidate-quota-v1" env PYTHONPATH=src \
        GRADSCF_NATIVE_THREADS=8 "$python" "$root/harness/bounded.py" \
        --seconds 1800 "$output.process.json" -- "$python" -m pytest -q \
        tests/test_scf_orbital_basis_gradients.py \
        tests/test_orbital_force_mixed_ad.py tests/test_neural_scf_force_training.py \
        tests/integrals/test_native_force_mixed_ad.py tests/xc/test_custom_xc.py \
        --junitxml="$output.xml" > "$output.log" 2>&1
    exit
fi
if test "$stage" = threads; then
    export PYTHONPATH=candidate-quota-v1/src GRADSCF_NATIVE_THREADS=64
    output="results/phi-svp-worker-budget-$allocation"
    test ! -e "$output.jsonl"
    srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c64 \
        "$python" harness/bounded.py --seconds 600 "$output.process.json" -- \
        "$python" harness/jk-budget-v2.py phenanthrene-def2-svp.json \
        "$output.jsonl" --threads 64 --workers 8 32 64 --omega 0. \
        --max-memory 60000 > "$output.log" 2>&1
    exit
fi
if test "$stage" = vjp; then
for variant in candidate-grid candidate-quota-v1; do
    export PYTHONPATH="$variant/src"
    output="results/phi-svp-integral-vjp-$variant-$allocation"
    test ! -e "$output.jsonl"
    srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c64 \
        "$python" harness/bounded.py --seconds 1200 "$output.process.json" -- \
        "$python" harness/jk-budget-v2.py phenanthrene-def2-svp.json \
        "$output.jsonl" --threads 32 --workers 32 --omega 0. \
        --geometry-vjp --exchange-fraction .25 > "$output.log" 2>&1
done
exit
fi

# Never start a stage whose full declared timeout cannot fit the live lease.
deadline=$(squeue -h -j "$allocation" -o '%e')
remaining=$(( $(date -d "$deadline" +%s) - $(date +%s) ))
if test "$remaining" -lt 6120; then
    echo 'Insufficient live allocation time for a fresh 6000-second stage.' >&2
    exit 2
fi

export PYTHONPATH=candidate-quota-v1/src
if test "$stage" = reference; then
output="results/phi-quota-reference-$allocation"
test ! -e "$output.jsonl"
srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c64 \
    "$python" harness/bounded.py --seconds 6000 "$output.process.json" -- \
    "$python" harness/run-quota-v1.py forward "$output.jsonl" \
    --case-input phenanthrene-def2-tzvp.json --xc pbe0 --method dft \
    --backend direct --grid 1 --grid-block-size 4096 --threads 32 \
    --restart results/phi-refreshed-blocked-scf.npz --screening 0. \
    --repeats 0 --checkpoint "$output.npz" > "$output.log" 2>&1
exit
fi

if test "$stage" = gradient; then
output="results/phi-quota-gradient-$allocation"
srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c64 \
    "$python" harness/bounded.py --seconds 6000 "$output.process.json" -- \
    "$python" harness/run-quota-v1.py geometry "$output.jsonl" \
    --case-input phenanthrene-def2-tzvp.json --xc pbe0 --method dft \
    --backend direct --grid 1 --grid-block-size 4096 --threads 32 \
    --reference-state "results/phi-quota-reference-$allocation.npz" \
    --forward reference --repeats 0 > "$output.log" 2>&1
exit
fi

# Each invocation evaluates one FD pair. Every endpoint is logged.
case "$stage" in
    fd-0.0003) step=0.0003 ;;
    fd-0.0001) step=0.0001 ;;
    *) echo 'Use parameters, threads, vjp, reference, gradient, fd-0.0003 or fd-0.0001.' >&2; exit 2 ;;
esac
    output="results/phi-quota-fd-$step-$allocation"
    srun --jobid="$allocation" --exact --gres=none -N1 -n1 -c64 \
        "$python" harness/bounded.py --seconds 6000 "$output.process.json" -- \
        "$python" harness/run-quota-v1.py geometry "$output.jsonl" \
        --case-input phenanthrene-def2-tzvp.json --xc pbe0 --method dft \
        --backend direct --grid 1 --grid-block-size 4096 --threads 32 \
        --reference-state "results/phi-quota-reference-$allocation.npz" \
        --gradient-record "results/phi-quota-gradient-$allocation.jsonl" \
        --forward scf --repeats 0 --fd-step "$step" > "$output.log" 2>&1
