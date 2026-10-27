# TD response execution acceptance

Run scientific commands inside an authorized Linux allocation, using float64
and a freshly built or verified matching libcint extension. All entry points
import the working tree through `PYTHONPATH=src`. An existing allocation ID
must be checked live; the historical job number is not a reusable allocation.

Smoke (two independent SCF/TD objects plus original-operator certification):

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python benchmarks/probes/tddft_execution/run.py \
  reports/tddft_execution_fix/20261027-d67ce6c/matrix-smoke.json \
  /tmp/td-execution-smoke --seconds 300
```

Main final-source CPU results (first plus three new-object warm executions):

```bash
PYTHONPATH=src python benchmarks/probes/tddft_execution/run.py \
  reports/tddft_execution_fix/20261027-d67ce6c/matrix-final-verified.json \
  /tmp/td-execution-final --seconds 1200
```

Run `matrix-parent.json` from a separate d67ce6c source tree with this harness
copied into `benchmarks/probes/tddft_execution/`. Keep the old production
sources unchanged. `matrix-parent-cpu8.json` and `matrix-final-cpu8.json` are
the matched benzene multithread pair. CPU8 means eight logical CPUs (four
physical cores in this allocation), not eight physical cores.

`run.py` creates fresh subprocesses and enforces process/solve/RSS budgets.
It controls affinity, OMP/BLAS/native threads and cache/preallocation settings.
Every output directory must be new. It synchronizes results and separately
marks the clean phase before residual certification. The 50 ms RSS trace is
sampled process-group RSS; Linux `ru_maxrss` is a different process high-water
metric. GPU allocator peaks, pools and XLA estimates are separate quantities.

`worker.py` optionally records compiler and solver events in independent
diagnostic runs. These are not clean performance results. Solver callbacks
give true submitted/effective widths and purposes; their host intervals do
not reliably bracket device execution or XLA scheduling, even on CPU. Do not
sum them or infer an action-time fraction from them.

`kernel.py` measures synthetic s4 contractions; it does not generate Gaussian
integrals. `action.py` measures a prepared molecular operator at the observed
6/12 column widths, using a common saved reference. Both exclude preparation
from warm action timing, and neither is an end-to-end chemistry result.

`certify_saved.py` rebinds saved restricted states to an independently imported
parent operator. It checks root residuals and, optionally, AO-metric subspaces.
The comparison state file and its producer are explicitly recorded in the
report; orbitals are never substituted into the end-to-end runs.

Rebuild summaries without changing original JSONL files:

```bash
python benchmarks/probes/tddft_execution/accept.py EVIDENCE_DIR OUTPUT_DIR
python benchmarks/probes/tddft_execution/summarize.py EVIDENCE_DIR OUTPUT_DIR
```

`accept.py` performs post-run numeric gates before paired ratios. Generic raw
kernel process records have no eigenvalue certificate and therefore retain
unknown scientific status; their separate saved-array gate is in
`acceptance.json`. The ratio is always candidate time divided by parent time.
