#!/bin/bash
# Narrow final regression, serial after the large timing/FD job; no GPU needed.
#SBATCH --partition=home
#SBATCH --job-name=hybrid-final-tests
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:35:00
#SBATCH --output=final-tests-%j.log
set -u
cd "$SLURM_SUBMIT_DIR"
python=../gradscf-libcint-nnao-20261026/env-compare/bin/python
export PYTHONPATH=candidate-grid/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export JAX_ENABLE_COMPILATION_CACHE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=1
srun --exact --cpu-bind=threads -N1 -n1 -c1 "$python" harness/environment.py \
    results/phi-grid-gradient.manifest.json results/environment-final.json
srun --exact --cpu-bind=threads -N1 -n1 -c1 "$python" harness/bounded.py \
    --seconds 1800 results/final-response-regression.process.json -- "$python" -m pytest -q \
    candidate-grid/tests/test_molecular_reference_forward.py \
    candidate-grid/tests/test_scf_orbital_basis_gradients.py \
    candidate-grid/tests/test_orbital_force_mixed_ad.py \
    candidate-grid/tests/test_orbital_variational_fock.py \
    candidate-grid/tests/test_orbital_memory_budget.py::test_molecular_solver_relays_original_mb_and_seed_owners \
    --junitxml=results/final-response-regression.xml \
    > results/final-response-regression.log 2>&1
status=$?
echo "$status" > results/final-response-regression.exit
exit "$status"
