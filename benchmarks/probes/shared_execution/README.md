# Shared physical fields and response program reuse

This bounded validation reuses the existing response-grid3 and workflow probes;
all molecular physics is provided by `src/gradscf`. It compares clean parent
`c5c8b28e` with the three production commits recorded in the report. The saved
queue is a site-specific job controller, not a new production workflow API.
Use the cluster's current manual and a single existing allocation. Never run a
second allocation just to execute these commands concurrently.

The measured resource configuration was one NV5090, 16 logical CPUs, 128 GiB,
with heavy cases serial. GPU molecular cases use eight logical CPUs, native8,
BLAS/OpenMP1; small CPU downstream cases use CPU1/native1. CPU-only regressions
may use eight allocated logical CPUs. The report does not call these eight
physical cores or claim independent CPU8 timing coverage.

A narrow smoke command, inside the allocated environment:

```sh
srun -n1 -c8 --cpu-bind=cores env PYTHONPATH=src:tests JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=8 python -m pytest -q tests/test_restricted_response_math.py tests/test_direct_td_program_reuse.py tests/test_grid_field_execution_reuse.py tests/test_direct_td_policy.py
```

A main paired reproduction (fresh output directories, same native binary,
explicit `parent-src/` exported from the recorded parent, current `src/` from
this branch; do not switch or modify the user's original checkout):

```sh
for variant in parent candidate; do
  source_dir=src
  if test "$variant" = parent; then source_dir=parent-src; fi
  srun -n1 -c8 --cpu-bind=cores env PYTHONPATH="$source_dir:.:tests" JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 JAX_ENABLE_COMPILATION_CACHE=false XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=8 python -m benchmarks.probes.response_grid3.guard --seconds 900 --rss-bytes 107374182400 "results/$variant-benzene-ri.process.json" -- python benchmarks/probes/response_grid3/tddft.py benchmarks/probes/shared_execution/benzene_ri.json "results/$variant-benzene-ri"
done
```

`water.json` and `benzene.json` specify conventional direct full TDDFT;
`benzene_ri.json` specifies explicit auxiliary RI. Each contains three clean
fresh-object workflows. `benzene_compile.json` enables separate compilation
instrumentation with two workflows and must not be used as clean timing.
They keep full orbital spaces and grid level3. Process guard includes startup
and all repeats; configuration `process_budget_seconds` records that scope.

After copying the saved results to `REPORT/raw/`, metadata processing uses only
Python's standard library and may run locally:

```sh
python3 benchmarks/probes/shared_execution/summarize.py reports/shared_execution/20261110-c5c8b28e
```

See `../workflow_reuse/README.md` for independent six-facade and four-GW/BSE
reproduction, including capabilities and derivative regressions. The final
report distinguishes complete workflows from unsupported guards, isolated
passes from the retained compiler crash, program reuse from numeric caching,
and sampled RSS from allocator/cumulative counters. No automatic merge/push or
production approximation changes are part of this probe.
