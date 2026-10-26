# Minimal reproduction commands

Run numerical commands inside a **new** approved Slurm allocation on jy_gpu_2,
after reading the current cluster manual. Allocation109318 has ended. Use the
recorded Python/JAX versions, CPU float64 and this branch; do not repoint the
original checkout's editable installation. The exact historical inputs live
under `evidence/`, and all mathematical implementations stay in `src/`.

## Production build and narrow acceptance

From the isolated source root, with the selected environment active:

```bash
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_ENABLE_COMPILATION_CACHE=false
python -m gradscf.integrals._native.build \
  --build-dir ../build-libcint-repro --jobs 8 \
  --blas-library /lib64/libblas.so.3
python -m pytest -q tests/integrals/test_geometry_first_call.py \
  tests/integrals/test_eri_basis_symmetry.py \
  tests/integrals/test_native_packed_geometry.py
```

The build requires at least eight allocated logical CPUs for this `--jobs`
setting. Call-count nodes skip in the ordinary production build. A separate
source/library build with `--diagnostics` and
`GRADSCF_EXPECT_DIAGNOSTICS=1` runs those nodes; never use its times as clean
speed samples. Rebuilding a used path requires a fresh Python process.

## A synchronized fixed-input micro-product

```bash
task_cpu1=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
taskset -c "$task_cpu1" python tools/libcint_migration/products.py \
  --kind coefficient_hessian_vjp --basis def2-svp \
  --layout full --threads 1 --output ../repro-c-hvp.json
```

This records one lower/compile/execute and three warm calls, plus source/library
provenance after sampling. It writes full numerical inputs/results to a sibling
NPZ. `--kind` also selects first coefficient/exponent and coordinate products;
coordinate products accept `s4/s8`. These are bare fixed-input products, not
relaxed molecular energy derivatives or raw-exponent normalization totals.

## Independent public PBE-TDA workflow

```bash
taskset -c "$task_cpu1" python tools/libcint_migration/worker.py \
  reproducibility/libcint_migration/20261026-current/evidence/s9/clean-v3/formal-after/cases/s9v3-after-water-pbe_tda-gradscf-cpu1/input.json \
  ../repro-water-tda
```

Use the corresponding PySCF input from the same `cases/` directory for the
independent reference. Compare actual basis arrays, Cartesian AO count, grid,
state count, convergence and numerical gates before forming a time ratio.
`probes.py --list` gives the scoped representative method catalogue. Its
validation fixtures include FD and other checks and are not clean method timers.

## Installed package / GPU / NNAO

`no_pyscf_smoke.py` must run outside source imports, with `PYTHONPATH` unset in
the production environment. To repeat the installed first-request group, run
`test_geometry_first_call.py` with that interpreter and
`GRADSCF_TEST_INSTALLED=1`; child processes assert no PySCF and an installed
package prefix. Ordinary source testing leaves that flag unset.

The allocated GPU uses `JAX_PLATFORMS=cuda,cpu`; preserve Slurm's device mapping.
Set `GRADSCF_TEST_GPU=1` for `tests/mp/test_native_gpu_geometry.py`.
`nnao_step.py --device gpu --mode full-step --output <name>.json` reproduces the
small random-MACE trajectory. Run `--mode components` in another process:
synchronized diagnostic phases are not clean full-step timings. Historical
trained checkpoint/campaign accuracy and NNAO force supervision are outside
the validated fixture scope.

The before symmetry implementation is commit2d965a8. Recreate its numerical
sources in a separate comparison directory, build its own extension, and copy
only the current thin measurement tools and private `diagnostics.py` inspector.
Do not apply production optimization patches to that baseline. Before/after
source trees, actual libraries and numeric input hashes must remain recorded.
