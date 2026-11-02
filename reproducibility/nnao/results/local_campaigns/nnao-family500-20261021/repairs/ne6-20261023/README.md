# Ne6 clustered-curvature repair — 2026-10-23

> DZP jobs were superseded after a later Kr3 iterative-residual failure:
> [2026-10-24 bounded dense recovery and current job IDs](../kr3-20261024/README.md).
> The Ne6 validation below remains evidence for the earlier source revision.

## Cause and minimal change

The S-DZP run stopped at accepted update 442 while evaluating
`GMTKN55-v2/RG18/ne6` for update 443. SCF had converged; the stability
Davidson solve had not resolved all three requested low roots to 1e-7.
The 84-AO / 30-occupied problem has a 1620-dimensional orbital Hessian,
outside the existing 1024-dimensional dense-rescue budget.

Replaying the exact saved parameters reproduced the rejection on a V100.
An independent dense orbital-Hessian calculation gave the lowest curvatures
10.680745769151443, 10.681311287581012 and 10.681311287581039: positive,
with two essentially degenerate roots. This is not evidence of a physical
negative-curvature instability. The diagnostic materialized only this two-index
orbital Hessian (about 20 MiB), never a four-index ERI.

The sole numerical source change strengthens the existing one-time Davidson
retry: maximum iterations 300→500, subspace 80→160, initial directions 8→16.
The first attempt, 1e-7 residual threshold, stability criterion, bounded dense
fallback and rejection of unresolved/negative cases are preserved. Energy,
Fock assembly and the implicit SCF response equations are unchanged.

## Verified evidence

- The real Ne6 regression failed before the patch and passed afterward.
- Focused local CPU regression: **16 passed, 3 deselected, 36.32 s**. No finite
  differences were run. The unrelated facade case and two FD/native-contracted
  cases were outside this check.
- Remote exact-parameter reproduction: job **104436**, V100, JAX 0.10.0;
  original fused stability residual 2.162e-5, unresolved, nonfinite gradient.
- Patched full forward/implicit-backward/MACE pullback: job **104454**, V100;
  residual **5.900e-8**, `stable=True`, `stability_resolved=True`, finite
  coefficient and full-network gradients. Energy remained
  **-769.6888005924052 Eh**. This finite-gradient check is not a new FD validation.
- [Exact reproduction](reproduction.json), [patched GPU validation](validation.json),
  [checkpoint migration](deployment.json), [submission](submitted.json),
  [scheduler verification](queue-verification.json).

CPU command, from the GradSCF root:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
GRADSCF_NATIVE_THREADS=4 PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q \
tests/nnao/test_ne6_stability.py tests/test_df_stability_dense_rescue.py \
tests/test_df_rhf_stability.py tests/nnao/test_df_rhf_solver.py \
-k 'not facade and not finite_differences and not native_contracted'
```

## Deployment and continuation

The running SZP/TZP lanes retain the frozen original source, hash
`1d249956d1e90296fe83aa697453c8b849d8a26bcceafbfc7bd01a909b52fee5`.
DZP uses an isolated copy under remote `source/` with exactly one changed file,
`gradscf/scf/stability.py`; source hash
`1b5cef38a4fb336b5f3c0bda987c1af534ad2592cc835ca798089173acb83da6`.
Unrelated local UKS/grid edits were excluded from the server patch.

S-DZP update 442 and M-DZP update 16 were archived and explicitly migrated:
only the checkpoint identity's source hash changed. Parameter, gradient and Adam
NPY payloads were verified byte-identical; batch cursors and online accumulators
were retained. M-DZP resumes update 16, and L-DZP retains the original seed.
No initial/epoch replay, finite differences or additional training evaluations
were added. The failed uncommitted batch is evaluated again on resume.

| New job | Family | Start | Dependency |
|---|---|---:|---|
| 104461 | S-DZP | update 442 | none |
| 104462 | M-DZP | update 16 | afterok:104461 |
| 104463 | L-DZP | original initialization | afterok:104462 |

The blocked old M/L-DZP jobs 104094/104095 were cancelled after replacements
were submitted. All new jobs retain Graph-off, one A100_80G, 32 CPUs, 512 GiB,
72-hour allocations and the existing time-limit requeue handler. Batch 50,
Adam/learning-rate schedule, 1000-epoch target, fixed exponent pools and the
rolling 10-change / 0.1-mEh stopping rule remain unchanged.

At **2026-10-23 21:46 +08:00**, S-DZP was pending resources; M/L-DZP were
pending dependencies. The A100 training continuation has therefore not yet
been validated past update 443. The exact-case V100 gate above is complete;
it is not an all-family or indefinite-training convergence guarantee.

Remote campaign root:
`/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021/`.
Canonical per-family training outputs remain under `runs/<family>/train/`.
Original checkpoint archives and the isolated server source remain remote;
the numerical reports, launch scripts and reviewable patch are also saved locally.
