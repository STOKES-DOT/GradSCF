# TZP Ne6 recovery and A100 allocation — 2026-10-24

> Update, 2026-10-24 14:08 +08:00: S-TZP completed normally by the approved
> energy stop at epoch 63 / update 630. Its last online mean was
> -886.7480085328617 Eh and the 10-change sum was 0.0643221 mEh.
> [Completion record](early-stop-epoch63.json). M-TZP 104597 is eligible and
> waiting for A100 resources while DZP runs on the same node class.

## Failure and verification

S-TZP job 104096 stopped at accepted update 435, epoch 44 batch 6, on
`GMTKN55-v2/RG18/ne6`. SCF converged, but the original stability checker returned
an unresolved low-root residual of 7.37849e-7 against the unchanged 1e-7 limit.

This recovery reuses the one-file stability repair already validated for DZP;
it introduces no further change to the scientific solver. Source SHA256:
`1b5cef38a4fb336b5f3c0bda987c1af534ad2592cc835ca798089173acb83da6`.
The shared isolated source is under the preceding DZP repair directory.

Exact TZP failure parameters were checked by job **104593** on a V100:

- Stable and resolved, with eigen-residual **5.48621e-8**.
- Energy **-769.7227147810725 Eh**, within 4e-13 Eh of the original value.
- Finite implicit coefficient gradient and full MACE parameter gradient.
- No finite differences, skipped structures or weakened convergence criteria.

[GPU validation](validation.json), [checkpoint migration](deployment.json),
[submitted jobs](submitted.json), [scheduler verification](queue-verification.json).

## Restarted TZP lane

| Family | New job | Seed | Dependency |
|---|---:|---:|---|
| S-TZP | 104596 | update 435 | successful validation 104593 |
| M-TZP | 104597 | original initialization | afterok:104596 |
| L-TZP | 104598 | original initialization | afterok:104597 |

As requested, every formal TZP job now requests **one A100_80G, 512 GiB RAM,
32 CPUs, 72 hours**, retaining the existing checkpoint/time-limit requeue handler.
All three use Graph-off. Their original batch-50 Adam schedule, fixed exponent
pools, 1000-epoch target and rolling 10-change / 0.1-mEh stop remain unchanged.

Only the source identity of the update-435 checkpoint was migrated. Parameter,
gradient and Adam NPY payloads were checked byte-identical; batch cursor 5 and
online accumulators were retained. The original checkpoint is archived remotely.
The old blocked M/L-TZP jobs 104097/104099 were replaced after new submissions.

The earlier A800 validation request 104556 never started and was cancelled.
Its initial 512-GiB request was first reduced to 128 GiB, then the test moved to
an available V100. This single-case validation allocation is separate from the
formal A100/512-GiB training request.

## Scheduler boundary

At the last 2026-10-24 01:00 check, S-TZP was **pending**; the scheduler reported
`ReqNodeNotAvail`. No new TZP training step is claimed. The A100_80G node n001
was busy and cannot fit two 512-GiB allocations in its approximately 1007 GiB
of configured memory, so the requested DZP/TZP allocations may serialize.
No smaller training-memory budget was substituted.

S-DZP 104461 had already advanced past the original failed update 443 to
update 446 on A100. S-SZP 104091 continued, with 111 completed epochs and
learning rate 5e-4. Neither healthy training job was stopped by this recovery.

Canonical progress remains under `runs/<family>/train/`; launches and execution
records for this TZP recovery are in the current directory. All ongoing job IDs
are also indexed in `../../restarts/graph-off-20261023/submitted.json`.
