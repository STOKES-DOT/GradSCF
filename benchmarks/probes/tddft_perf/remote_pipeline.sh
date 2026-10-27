#!/bin/bash
# Run inside the single authorized Slurm allocation. Heavy probes are serial.
set -u
cd "$1"
while [[ ! -f ../baseline.exit ]]; do sleep 5; done
tar -xf ../stage2.tar
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin="$2"
result_dir=reports/tddft_perf/20261027-33a7cab
# Environment is collected outside every scientific timing interval.
"$python_bin" tools/libcint_migration/environment.py "$result_dir/environment.json" > ../environment.log 2>&1
for matrix in hybrid-small ad kernels diagnostics experiments actions molgw; do
    remaining=$("$python_bin" -c 'import os,time;print(max(0,int(7200-(time.time()-os.stat("reports/tddft_perf/20261027-33a7cab/runs/water-pbe-tda-gradscf-cpu1-clean-e2e/input.json").st_mtime))))')
    # Also record each phase's available wall budget, before entering it.
    printf '%s %s\n' "$matrix" "$remaining" >> ../phase-budgets.txt
    if (( remaining < 20 )); then break; fi
    "$python_bin" benchmarks/probes/tddft_perf/run.py "$result_dir/matrix-${matrix}.json" "$result_dir" --seconds "$remaining" > "../${matrix}-driver.log" 2>&1
done
printf '%s\n' "$?" > ../pipeline.exit
