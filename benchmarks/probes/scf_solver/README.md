# Shared SCF solver acceptance

The catalog freezes60 physical systems, five balanced bases and four XC classes.
See [FIXTURE_CATALOG.md](FIXTURE_CATALOG.md) for input provenance and constructed
geometries. All numerical commands below run inside a Slurm allocation. Metadata
generation and summary commands require only Python's standard library.

## Reproduce inputs and run one worker

From the tested source checkout:

```bash
python benchmarks/probes/scf_solver/make_plan.py /new/path/scf-plan
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python benchmarks/probes/scf_solver/runner.py \
  /new/path/scf-plan/configs/methane-cc-pvdz-rhf-hf-gradscf.json \
  /new/path/methane-rhf-result
```

The worker requires `SLURM_JOB_ID`, enforces the configured CPU affinity and
records actual import/native paths. Use a fresh output directory. The SCF facade
timer excludes imports, basis-dimension preflight and final certification/state
I/O; the process guard covers the whole worker. Registered primary cases have
one call; selected controls have one first call and three new-object warm calls.

## Execute or resume the complete registered matrix

`campaign.py` is a metadata-only controller. It launches one independent guarded
`srun` step at a time inside an existing allocation; it never allocates resources.
Its campaign root expects the existing comparison environment at
`../gradscf-libcint-nnao-20261026/env-compare/bin/python`. Set up that environment
explicitly or adapt that one environment path for another installation.

```bash
python benchmarks/probes/scf_solver/campaign.py \
  /remote/campaign /remote/tested-source /remote/scf-plan ALLOCATION_ID \
  --worker /remote/frozen-harness/runner.py
```

Each step requests240GiB and either8 or64 logical CPUs. This is a serial acceptance
protocol, not eight simultaneous CPU8 jobs. A completed `.process.json` prevents
rerunning or overwriting an attempt; an orphan log requires inspection. A lease
too short for the registered per-process upper bound defers the case. The default
release marker ends the campaign's hold allocation; use `--keep-allocation` only
when a following explicitly controlled phase owns release. `--case-list` accepts
a JSON list of existing IDs for separate correction attempts, without modifying
the frozen configurations. Give correction attempts a distinct `--output-name`.

## Recover NFS metadata and summarize

```bash
python benchmarks/probes/scf_solver/refresh_results.py \
  /remote/scf-plan /remote/results/final-matrix /remote/tested-source ALLOCATION_ID
python benchmarks/probes/scf_solver/summarize.py \
  /remote/scf-plan /remote/results/final-matrix /remote/summary \
  --overlay /remote/results/corrections \
  --timing-exclusions /remote/timing-exclusions.json
```

The optional overlay selects explicit correction attempts and retains provenance
and `supersedes_run`; raw original outputs stay untouched. Timing exclusions are
a JSON list of IDs measured alongside diagnostics. Numerical acceptance remains
separate from timing acceptance. Missing, unsupported, incomplete, unconverged
or mismatched runs never produce a speed ratio. Ratios are GradSCF/PySCF, with
different conventional execution strategies labeled explicitly. No mixed-method
average is reported.

`watch_summary.py` is the specific20261104 remote recovery watcher. It reads the
frozen candidate/harness layout documented in the report, rebuilds derived status
every300s, and exits when the allocation leaves the queue. It performs no quantum
calculation or job submission. Its dated paths deliberately identify that run;
it is not a generic cluster management layer.

## Narrow regressions

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
  tests/solvers/test_scf.py tests/solvers/test_scf_orthogonalization.py \
  tests/test_scf_hermitian_roundoff.py tests/test_scf_ghf_gks.py \
  tests/test_scf_campaign_records.py tests/test_scf_summary_overlay.py
```

Training, force-loss and NNAO checks use the separately recorded nodes in the
implementation report. The small MACE fixture does not validate a trained model
checkpoint. No numerical GPU pass follows from obtaining a GPU allocation.
