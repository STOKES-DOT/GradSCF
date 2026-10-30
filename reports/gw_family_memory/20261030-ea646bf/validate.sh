#!/bin/bash
# Final affected-path checks, isolated by file to keep compiler/RSS scopes clear.
set -u
cd "$HOME/private/homefile/gradscf-gw-family-memory-20261030"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH="$PWD/candidate/src"
PY=../gradscf-libcint-nnao-20261026/env-compare/bin/python
check() {
    name=$1
    file=$2
    shift 2
    timeout 1200 "$PY" -m pytest -q "candidate/tests/gw/$file.py" "$@" --tb=short \
        -o junit_family=xunit1 --junitxml="results/final-$name.xml" >"results/final-$name.log" 2>&1
    code=$?
    echo "$code" >"results/final-$name.exit"
    tail -20 "results/final-$name.log"
    return "$code"
}
for group in test_evgw_preparation test_conventional test_qsgw_streaming test_gw_qsgw test_scgw_working_arrays test_gw_scgw; do
    check "$group" "$group" || exit $?
done
check qsgw-outer test_gw_outer_response -k qsgw || exit $?
check scgw-response test_gw_scgw_response -k 'compiled_primal or uniform_energy_shift or implicit_response_matches or noninteracting_degenerate or h2_df' || exit $?
check nnao test_nnao_qp_training || exit $?
echo 0 >results/final-validation.exit
