#!/bin/bash
# Run only inside a current allocation, from this isolated project's directory.
set -u
export PATH="$PWD/env-compare/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$PWD/source-v3/src"
cd source-v3 || exit 1
timeout 6000 python -m pytest -q \
  tests/test_public_import_defaults.py tests/test_solver_compile_reuse.py \
  tests/test_grid_aligned_end_to_end_tddft.py tests/integrals/test_native_ri.py \
  --junitxml=../evidence/final-core.xml --tb=short \
  >../evidence/final-core.log 2>&1
echo "$?" >../evidence/final-core.exit
tail -25 ../evidence/final-core.log
timeout 6000 python -m pytest -q tests/test_molecular_correlated_response.py \
  -k 'not mixed_raman and not nitrogen' \
  --junitxml=../evidence/final-correlated.xml --tb=short \
  >../evidence/final-correlated.log 2>&1
echo "$?" >../evidence/final-correlated.exit
tail -25 ../evidence/final-correlated.log
cd .. || exit 1
export PYTHONPATH="$PWD/baseline/src"
timeout 6350 python source-v3/tools/libcint_migration/run_cases.py \
  gradscf-libcint-ri-gwbse.json evidence/ri-gwbse-before --seconds 6300 \
  >evidence/ri-gwbse-before.log 2>&1
echo "$?" >evidence/ri-gwbse-before.exit
tail -5 evidence/ri-gwbse-before.log
export PYTHONPATH="$PWD/source-v3/src"
timeout 6350 python source-v3/tools/libcint_migration/run_cases.py \
  gradscf-libcint-ri-gwbse.json evidence/ri-gwbse-after --seconds 6300 \
  >evidence/ri-gwbse-after.log 2>&1
echo "$?" >evidence/ri-gwbse-after.exit
tail -5 evidence/ri-gwbse-after.log
timeout 6350 python source-v3/tools/libcint_migration/run_cases.py \
  gradscf-libcint-final-cpu1.json evidence/after-final --seconds 6300 \
  >evidence/after-final.log 2>&1
echo "$?" >evidence/after-final.exit
tail -5 evidence/after-final.log
timeout 6350 python source-v3/tools/libcint_migration/run_cases.py \
  gradscf-libcint-cpu8.json evidence/cpu8-after --seconds 6300 \
  >evidence/cpu8-after.log 2>&1
echo "$?" >evidence/cpu8-after.exit
tail -5 evidence/cpu8-after.log
