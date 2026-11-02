# Eight-family training restart — CUDA Graph disabled

> DZP queue updated on 2026-10-23 after the Ne6 stability repair:
> [current DZP jobs and checkpoint lineage](../../repairs/ne6-20261023/README.md).
> The table below retains the original 10:10 submission snapshot; use the
> updated `submitted.json` for current DZP job IDs.

> TZP queue updated on 2026-10-24: [Ne6 recovery and A100/512-GiB jobs](../../repairs/ne6-tzp-20261023/README.md).
> The latest `submitted.json` includes these replacement TZP jobs as well.

> Latest DZP queue update, 2026-10-24: [Kr3 bounded dense recovery](../../repairs/kr3-20261024/README.md).
> Active DZP jobs are now 104927 → 104928 → 104929.

Snapshot: **2026-10-23 10:10 +08:00**. All eight jobs were submitted and their
actual Slurm scripts/resources/dependencies checked. The three lane heads are
pending resources/priority; no new numerical training step is claimed yet.

| Lane | Serial job chain | Starting accepted updates |
|---|---|---|
| SZP | S `104091` → M `104092` | 0, 0 |
| DZP | S `104093` → M `104094` → L `104095` | 16, 16, 0 |
| TZP | S `104096` → M `104097` → L `104099` | 40, 0, 0 |

## Runtime correction and preserved training definition

Every job sets `--xla_gpu_enable_command_buffer=` before JAX initialization.
The frozen source SHA256 remains
`1d249956d1e90296fe83aa697453c8b849d8a26bcceafbfc7bd01a909b52fee5`.
Original source, inputs and workflow files remain unchanged; these new wrappers
invoke the existing campaign and checkpoint loader.

Preserved: 500 structures, fixed exponent candidates, original MACE/seed 0,
float64 DF-RHF, implicit backward tolerance 1e-8, batch 50, micro-batch 1,
Adam 1e-3 with halving every 100 epochs, maximum 1000 epochs, and the rolling
10-change absolute online-mean energy stop threshold of 1e-4 Eh. No finite
differences, proposal scoring or additional SCF replay was introduced.

S-DZP and M-DZP retain update-16 parameters, Adam state, batch cursor and online
accumulators. The completed graph-off S-TZP update-40 checkpoint was promoted
only after matching its saved hash and original checkpoint identity. Old
checkpoints and progress/metadata are preserved under remote `before/`.
The other five families retain their original initialization, including L-DZP,
which had not completed an update before cancellation.

## Resources and continuation

At most three campaign GPUs run concurrently: one serial lane per zeta group.
Each job requests 32 CPUs. SZP uses n005/A800 with 448 GiB RAM; TZP uses
n005/A800 with 512 GiB; DZP uses A100_80G with 512 GiB.

n005 enforces an 8-hour allocation limit; A100 jobs request 72 hours. Five minutes
before each allocation limit, a batch-shell USR1 handler requeues the same job.
The existing atomic `last.npz` preserves the most recently committed batch; a
partially evaluated batch can be repeated after restart. This does not reserve
the same physical GPU across allocations. Logs append, and execution records
include `SLURM_RESTART_COUNT`.

Success dependencies use `afterok`, so a genuine training failure stops that
lane rather than automatically launching the next family. Manual user stops
are not automatically requeued by the time-limit handler.

## Verification and limits

- [Plan and checkpoint lineage](plan.json), [preflight](preflight.json).
- [Submitted job IDs and exact script hashes](submitted.json).
- [Verification of all eight actual Slurm job records](verification.json).
- [Three-lane Slurm test-only acceptance](submission-dry-run.json).
- Eight shell syntax checks and three local mock cases passed: normal success,
  numerical failure and USR1 requeue. No scientific calculation was run by them.
- Real cross-allocation requeue remains to be observed; the local test does not
  prove that a long training run or GPU-memory residency is indefinitely stable.

The preceding bounded S-TZP control resumed update 16 → 40 without OOM, adding
24 updates, and stopped normally at the prescribed budget. It recorded a maximum
sampled GPU footprint of 63901 MiB and process peak RSS of 191323888 KiB.
This is not a same-hardware speed comparison or a proof that all families fit.

Remote root:
`/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/`.
Canonical progress remains in `artifacts/nnao-family500-20261021/runs/<family>/train/`;
this restart's launch/execution/audit records are under `restarts/graph-off-20261023/`.
