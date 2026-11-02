set -eu
cd /tmp/gradscf-szp663-20260916
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL
export PYTHONPATH=src
py=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
cp -a artifacts/methane-szp663 artifacts/methane-szp663-160steps
taskset -c 0-3 "$py" gradscf-szp663-run.py optimize --basis-family szp663_direct --initial-summary artifacts/methane-core6/summary.json --maxiter 600 --output-dir artifacts/methane-szp663 > artifacts/methane-szp663/optimization.log 2>&1 || { tail -60 artifacts/methane-szp663/optimization.log; exit 1; }
taskset -c 0-3 "$py" tests/comparisons/validate_methane_nnao.py artifacts/methane-szp663/summary.json > artifacts/methane-szp663/reference.log 2>&1
taskset -c 0-3 "$py" tools/compare_methane_core6_basis_sets.py artifacts/methane-szp663/summary.json --bases NNAO-core6-6s6p3d-H6s2p --output artifacts/methane-szp663/comparison > artifacts/methane-szp663/comparison.log 2>&1
taskset -c 0-3 "$py" tests/comparisons/validate_methane_core6_basis_sets.py artifacts/methane-szp663/comparison/comparison.json artifacts/methane-szp663/summary.json > artifacts/methane-szp663/comparison/reference.log 2>&1
tail -7 artifacts/methane-szp663/optimization.log
cat artifacts/methane-szp663/reference.log
cat artifacts/methane-szp663/comparison.log
