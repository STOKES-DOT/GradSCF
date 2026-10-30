# Molecular G0W0 memory probes

Run numerical work inside an authorized Slurm allocation, with this checkout's
`src` first on `PYTHONPATH`. The production native library must match its C/C++
sources. The recorded run reused the identical isolated libcint library; no
production integral or SCF kernel was modified.

```sh
export PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=1 JAX_ENABLE_COMPILATION_CACHE=false
python benchmarks/probes/g0w0_memory/run.py residue /tmp/gw-residue.jsonl --naux 128 --repeats 10
python benchmarks/probes/g0w0_memory/run.py water /tmp/gw-water.jsonl --nw 32 --repeats 3 --threads 1 --device cpu
```

The worker sets CPU affinity **before importing JAX**, synchronizes the whole
result, and uses a new SCF facade each repeat. Repeat 0 includes first-use
compilation; later repeats reuse compiled programs, never a saved QP result.
Both factor preparation and SCF remain inside the molecular end-to-end timer.
Native RI/SCF preparation stays on CPU. `--device gpu` explicitly transfers the
six required physical arrays and records that transfer before GPU response.

Each molecular repetition has a 6000 s OS alarm. The outer controller also
needs a process/round deadline, distinct from this per-calculation deadline.
An interrupted later repeat does not mean that an earlier completed solve
failed. Inspect `scientific_pass`, every requested root's residual, positive Z,
branch flag and coverage, and the run-level process status separately.

`--case-input` accepts the saved JSON under `reports/g0w0_memory/*/cases/`.
The initial water protocol uses four QPs and 32 imaginary nodes. The benzene
and larger-basis water probes use six QPs and 64 nodes, with all G/W states.
These are separate workloads; their times must not be pooled into a scaling
exponent. An explicitly selected `--screening-solver gmres` uses the shared
checked implicit solver, with no dielectric/transition Gram matrix.

For the recorded CPU64 series, the allocation contains 64 logical CPUs and
BLAS/OpenMP use one thread to avoid nesting; libcint and XLA may use the
allocated CPU set. CPU8 means logical CPUs, not eight physical cores. The
worker records package/core identities for the actual affinity.

`residue-ad` times value plus parameter VJP. XLA temporary estimates and
Linux process peak RSS are separate metrics; RSS includes imports, native
preparation, compilation, allocator retention and execution. GPU allocator
bytes-in-use, peak and pool counters are recorded only when exposed; missing
counters remain null. No allocator pool snapshot is called an active peak.

`reference.py` starts PySCF or the explicitly supplied unmodified MOLGW binary
from the same geometry/basis configuration. MOLGW subprocesses have fresh
directories and no restart files or `MOLGW_COMPARE_*` environment. The records
deliberately leave cross-software `time_ratio` empty: their root diagnostics
and, for MOLGW, frequency algorithms differ from GradSCF's six-seed CD
certification. Compare numerical values and separately declared work first.

## Regression entry

```sh
python -m pytest -q tests/gw/test_projected_residue.py tests/gw/test_gw_memory_plan.py tests/gw/test_selected_factors.py tests/gw/test_selected_driver.py tests/gw/test_dynamic_screening_action.py tests/gw/test_independent_ri.py tests/gw/test_nnao_qp_training.py
```

The NNAO test is a synthetic four-parameter H2 radial fixture. It verifies
native contracted integrals/RI, responding overlap/density/orbitals, nonlinear
QP derivatives, two finite-difference steps, rebinding and an Adam update. It
does not certify a trained MACE checkpoint, full RI nuclear derivatives, or a
force-supervised NNAO campaign.
