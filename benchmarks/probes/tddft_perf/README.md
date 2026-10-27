# TDA / TDDFT performance probes

Frozen production baseline and scientific results:
[`reports/tddft_perf/20261027-33a7cab/report.md`](../../../reports/tddft_perf/20261027-33a7cab/report.md).
No production kernel is changed by this harness.

`run.py MATRIX OUTPUT --seconds 7200` serializes isolated child processes and
enforces per-case wall-time and sampled RSS guards. Each case records its full
input, actual command, stdout, RSS samples and measurements. The worker sets
its CPU affinity and OpenMP/BLAS/JAX platform before importing numerical code.
Run numerical probes only within an authorized Slurm allocation.

- `worker.py`: independent SCF-to-excitation flows, preparation and compiled
  action measurements; explicit CPU-preparation/GPU-response route is labeled.
- `shared_probe.py`: identical orbitals and grid for both response engines.
- `kernel_probe.py`: dynamic synthetic s4 / general raw-DF kernels; validation
  and oracle-free peak-memory runs are separate.
- `experiments.py`: three exact, process-local rearrangements; no approximation.
- `ad_probe.py`, `validation_probe.py`: existing geometry, neural-XC and NNAO
  consumers, JVP/VJP/FD and a small random-MACE optimizer fixture.
- `nto_probe.py`: offline SVD and K-action errors on actual Davidson vectors.
- `stage_probe.py`: separate integral, RI, XC and J/K microbenchmarks.
- `pyscf_convergence_probe.py`, `molgw_probe.py`: preserve external-reference
  failures and residuals; an assigned but unused PySCF 2.13 `max_space` property
  is not evidence of an effective subspace change.
- `analyze.py`, `make_report.py`: saved-data analysis only, standard library.

The main matrix uses explicitly registered float64/grid/SCF/response controls.
Some original probes inherited irrelevant metadata from their template; the
raw files remain, and `cases/*.json` effective controls plus harness-correction notes
state the actual fixture. Report generation for this campaign expects its
complete saved run set. A smaller fresh rerun can always use `analyze.py`.

`reproduce.sbatch` is a bounded full rerun on the same resource class. Configure
the existing Python path and optional MOLGW executable/runtime paths from
`commands.txt`; no dependency installation or source migration happens in the
batch script. Its nonconverged/unsupported cases are intentional diagnostics,
not successful performance comparisons. Use a fresh output directory.
