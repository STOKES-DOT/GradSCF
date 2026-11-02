set -eu
cd /tmp/gradscf-core6-comparison-20260916
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL
export PYTHONPATH=src
py=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
"$py" - <<'CHECK'
import hashlib,json,pathlib
m=json.loads(pathlib.Path('comparison-hashes.json').read_text())
bad=[p for p,h in m.items() if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h]
assert not bad,bad
print('Verified source files:',len(m),flush=True)
CHECK
mkdir -p artifacts/comparison10
taskset -c 0-3 "$py" -m pytest -q tests/integrals/test_pople_polarization_names.py tests/integrals/test_basis_library.py tests/test_no_external_chemistry_dependency.py > artifacts/comparison10/tests.log 2>&1
summary=artifacts/methane-core6/summary.json
taskset -c 0-3 "$py" tools/compare_methane_core6_basis_sets.py "$summary" --output artifacts/comparison10 > artifacts/comparison10/run.log 2>&1
taskset -c 0-3 "$py" tests/comparisons/validate_methane_core6_basis_sets.py artifacts/comparison10/comparison.json "$summary" > artifacts/comparison10/reference.log 2>&1
cat artifacts/comparison10/run.log
cat artifacts/comparison10/reference.log
