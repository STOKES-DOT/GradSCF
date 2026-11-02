# Kr3 stability recovery — 2026-10-24

## Failure and diagnosis

S-DZP 104461 stopped after update **470** / epoch 47. The next batch failed on
`GMTKN55-v2/RG18/kr3`: SCF converged, but its A100 low-root stability residual
was **1.058e-6**, exceeding the unchanged **1e-7** criterion.

The same saved parameters passed naturally on V100 (job **104912**), so a V100
failure reproduction is **not** claimed. Independent dense diagonalization of
the actual 1458-dimensional orbital Hessian gave minimum curvature
**5.852014827185982**, with residuals below 2.46e-12. The positive, clustered
low spectrum and hardware-sensitive iterative residual motivated a bounded
reference fallback, rather than accepting an unresolved result.

## Minimal correction

After the existing Davidson retry, reuse `EigenSolverConfig.max_dense` instead
of a separate hard-coded 1024 cutoff. Its default permits up to **2048** orbital
directions, or **32 MiB per float64 Hessian matrix**. Kr3's matrix is about
16.2 MiB. Columns are applied sequentially to bound DF intermediates.

This materializes only a temporary orbital Hessian, not a full four-index ERI.
Residual/stability tolerances, negative-mode escape, larger-space rejection,
energy/Fock formulas and implicit SCF response equations remain unchanged.

## Verification

- The expanded-budget regression failed before the correction.
- Focused CPU checks: **18 passed, 3 deselected, 41.54 s**. Includes the real
  Kr3 inputs and an injected iterative miss; no finite differences were run.
- V100 patched validation **104924**: natural residual 4.91e-10; explicitly
  forced iterative-miss/dense-rescue residual **2.07e-12**. Both energy and
  coefficient gradient agree, and the full MACE pullback is finite.
- Energy: **-8232.0341387244 Eh**. The forced miss is a control-flow test;
  its dense solve and gradient use the real physical inputs.

[Reproduction](reproduction.json), [GPU validation](validation.json),
[migration](deployment.json), [jobs](submitted.json),
[scheduler check](queue-verification.json).

CPU command from GradSCF root:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
GRADSCF_NATIVE_THREADS=4 PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q \
tests/nnao/test_kr3_stability.py tests/nnao/test_ne6_stability.py \
tests/test_df_stability_dense_rescue.py tests/test_df_rhf_stability.py \
tests/nnao/test_df_rhf_solver.py \
-k 'not facade and not finite_differences and not native_contracted'
```

## Deployment and continuation

Only DZP changes to isolated source SHA256
`7e5a22c4a17e3d727c03c121be4484e902080cef7b39b164783ce5fde23cdec2`.
The healthy M-SZP and TZP sources were not modified. Exactly one scientific
file differs from the prior `1b5cef...` source: `gradscf/scf/stability.py`.

S-DZP update 470 and M-DZP update 16 were explicitly migrated and archived;
only their source identity changed. Parameter, gradient and Adam NPY payloads
were verified byte-identical. Batch cursors, online accumulators and the original
seed/learning-rate/early-stop settings remain intact. The original failed
update-471 snapshot was also archived separately.

| New job | Family | Seed | Dependency |
|---|---|---:|---|
| 104927 | S-DZP | update 470 | successful validation 104924 |
| 104928 | M-DZP | update 16 | afterok:104927 |
| 104929 | L-DZP | original initialization | afterok:104928 |

All request one A100_80G, 32 CPUs, 512 GiB and 72 hours, retaining Graph-off
and time-limit requeue. Old queued jobs 104462/104463 were replaced. S-DZP
will run a **one-time exact-case check on its allocated A100 before training**,
saved as `a100-validation.json`; this is not an epoch-wide replay or an update.

At the 2026-10-24 11:40 check, the new S-DZP was pending resources. No post-fix
A100 training update is claimed yet. M-SZP had 38 completed epochs and TZP 49;
both were running. The updated overall job index remains at
`../../restarts/graph-off-20261023/submitted.json`.
