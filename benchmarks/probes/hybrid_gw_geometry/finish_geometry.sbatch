#!/bin/bash
# Run from the frozen remote campaign root, after its active gradient job.
#SBATCH --partition=home
#SBATCH --job-name=hybrid-geometry-accept
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:NV5090:1
#SBATCH --time=04:00:00
#SBATCH --output=geometry-accept-%j.log
set -u
cd "$SLURM_SUBMIT_DIR"
python=../gradscf-libcint-nnao-20261026/env-compare/bin/python
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=32
hostname
lscpu > "hardware-accept-$SLURM_JOB_ID.txt"

# Consume a completed gradient only; FD endpoints run fresh SCF solves.
if "$python" -c 'import json; from pathlib import Path; rows=[json.loads(x) for x in Path("results/phi-grid-gradient.jsonl").read_text().splitlines()]; assert any(r.get("stage")=="derivative" and r.get("scientific_pass") for r in rows)'; then
    export PYTHONPATH=candidate-grid/src
    for step in 3e-4 1e-4; do
        output="results/phi-grid-fd-$step"
        if test -e "$output.jsonl"; then
            echo "Refusing to overwrite $output.jsonl" >&2
            exit 2
        fi
        srun --exact --gres=none -N1 -n1 -c64 "$python" harness/bounded.py \
            --seconds 6000 "$output.process.json" -- "$python" harness/run-grid.py \
            geometry "$output.jsonl" --case-input phenanthrene-def2-tzvp.json \
            --reference-state results/phi-refreshed-blocked-scf.npz \
            --xc pbe0 --method dft --backend direct --grid 1 --grid-block-size 4096 \
            --threads 32 --screening 0 --forward scf --repeats 0 \
            --gradient-record results/phi-grid-gradient.jsonl --fd-step "$step" \
            > "$output.log" 2>&1
        echo "$?" > "$output.exit"
    done
else
    echo "No accepted large gradient: finite-difference branch was not run." >&2
fi

# Fully serial fixed-density comparison: no numerical regression shares SMT.
for variant in candidate-brillouin candidate-grid; do
    export PYTHONPATH="$variant/src"
    output="results/phi-fock-serial-$variant"
    if test -e "$output.json"; then
        echo "Refusing to overwrite $output.json" >&2
        exit 2
    fi
    srun --exact --gres=none -N1 -n1 -c64 "$python" harness/bounded.py \
        --seconds 6000 "$output.process.json" -- "$python" harness/stationarity.py \
        results/phi-tzvp-pbe0.npz phenanthrene-def2-tzvp.json \
        results/phi-tzvp-pbe0-g0w0-gpu.jsonl "$output.json" \
        --threads 32 --fock-repeats 3 > "$output.log" 2>&1
    echo "$?" > "$output.exit"
done
srun --exact --gres=none -N1 -n1 -c1 "$python" harness/compare_fock.py \
    results/phi-fock-serial-candidate-brillouin.json \
    results/phi-fock-serial-candidate-grid.json harness/fock_acceptance.json \
    results/phi-fock-serial-comparison.json
