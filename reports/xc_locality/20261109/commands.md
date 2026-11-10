# Reproduction

All numerical commands run inside an authorized Slurm allocation on jy_gpu_2,
not on the login node or local Mac. Recheck the cluster manual and current
allocation before use. One NV5090, 64 logical CPUs (32 physical on this node),
256 GiB; native 64, BLAS/OpenMP 1, float64. Run heavy cases serially.

The retained remote archive is:

```bash
ROOT=/home/user_yuanjiao/private/homefile/gradscf-xc-locality-20261109
BASE=/home/user_yuanjiao/private/homefile/gradscf-hfks-finish-20261106
PY=/home/user_yuanjiao/private/homefile/gradscf-libcint-nnao-20261026/env-compare/bin/python
REF="$BASE/representatives-v2/beta-pbe0-direct"
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/tests"
export JAX_ENABLE_X64=1 JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=64
cd "$ROOT"
```

The source-only baseline is `20a47391`; the candidate production commit is `db9700ff`.
`paired_ready_zero_weight_fix.json` records final candidate source hashes. The
earlier `paired_ready.json` and all original scripts/logs are historical records.
The original paired controller was paused between baseline and candidate while
the zero-weight review issue was fixed. Its numerical baseline step was not
signaled. Do not rerun its old receipt check unchanged against the final source.

## Smoke

Inside the allocated batch shell, with a new output directory:

```bash
srun -n1 -c64 --cpu-bind=cores "$PY" limit.py \
  -m benchmarks.probes.response_grid3.guard --seconds 1200 \
  --rss-bytes 214748364800 smoke-replay.process.json -- \
  "$PY" benchmarks/probes/scf_100/local_scf_benchmark.py \
  "$REF/gradscf/input.json" "$REF/gradscf/summary.json" smoke-replay \
  --device cpu --screen-tol 1e-14 --smoke
```

## Complete SCF comparison

Run this once per device and variant, using fresh output paths. Baseline uses
`SOURCE="$BASE/src"; CUTOFF=0`; candidate uses `SOURCE="$ROOT/src"; CUTOFF=1e-14`.
`DEVICE` is `cpu` or `gpu`; each process runs cold plus 3 fresh-input warm calls.

```bash
export PYTHONPATH="$SOURCE:$ROOT" GRADSCF_EXPECTED_SRC="$SOURCE"
srun -n1 -c64 --cpu-bind=cores "$PY" limit.py \
  -m benchmarks.probes.response_grid3.guard --seconds 6000 \
  --rss-bytes 214748364800 "$NEW_OUT.process.json" -- \
  "$PY" benchmarks/probes/scf_100/local_scf_benchmark.py \
  "$REF/gradscf/input.json" "$REF/gradscf/summary.json" "$NEW_OUT" \
  --device "$DEVICE" --screen-tol "$CUTOFF"
```

Keep the complete process receipt and all 4 scientific records. The ordinary
grid/JIT caches are allowed; no converged density or orbital restart is reused.
The independent per-process guard is 6000 s / 200 GiB. First-call timing starts
after module imports and native registration; process wall time is recorded
separately and includes all 4 calls.

## Fixed-density operators

`local_xc_benchmark.py CONFIG DENSITY GRID NEW_OUT --device cpu --selected-only
--cpu-cache` compares raw stream, selected local policy and an admitted CPU AO
cache. On GPU omit `--cpu-cache`. Inputs are the saved validated GradSCF density
and full grid at `$REF/gradscf/density.npz` and`$REF/shared-grid.npz`; hashes are
in each manifest. The geometry/basis/configuration is also in `cases/`.

`whole_mgga_benchmark.py CONFIG SCF_FINAL_STATE GRID NEW_OUT --device cpu|gpu`
compares the whole r2SCAN operator with/without hints at actual saved PBE0
C/D/occupations. It does not run r2SCAN SCF, and does not rebuild D to make the
strict hint guard pass.

## Rebuild the archived summary

This operation uses stdlib only and can run locally:

```bash
python3 benchmarks/probes/scf_100/summarize_xc_locality.py \
  reports/xc_locality/20261109/raw reports/xc_locality/20261109/results
```

Raw XML, logs, JSONL and source manifests remain under `raw/` locally and in the
remote archive; they are excluded from Git. The curated summary and inputs are
tracked. `harness_guard.patch` preserves the already-present monitoring fix used
in this run; it does not change any numerical kernel.

On a clean reproduction checkout, apply the recorded harness patch only if
`git apply --check` succeeds; the retained remote archive already contains it.
This changes exception cleanup/monitoring, not successful numerical kernels.
