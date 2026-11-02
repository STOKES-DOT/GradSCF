# Submission status — 2026-10-21

All eight formal training jobs have been submitted. Latest live check after
the A800 update: preparation job 102860 is PENDING (Resources); all formal
jobs are PENDING (Dependency). No formal epoch has started. Original
preparation 102758 failed before compilation at a nonexistent GCC12 enable
path; the replacement uses available system compiler commands.

| Lane | S | M | L |
| --- | --- | --- | --- |
| SZP | 102863 | 102869 | Existing L-SZP retained |
| DZP | 102870 | 102760 | 102761 |
| TZP | 102871 | 102765 | 102766 |

Every lane begins after successful preparation 102860; successors use
afterany on the preceding family within that lane. All jobs request one
GPU and 32 CPU. Preparation requests A800/n005/512 GiB for 4 h; SZP
requests A800/n004/448 GiB for 72 h. DZP/TZP retain A100_80G/512 GiB/72 h.
At most three family jobs can run at once; node memory and GPU availability
can reduce actual concurrency. Every dependency was checked, with no
held or DependencyNeverSatisfied job remaining in this campaign.

Preparation builds native integrals from the exported main-workspace
sources, runs the narrow test gate, calibrates the missing M pool, and runs
eight full-model H2 one-update smoke checks. These remote checks are pending.
S/L candidates already passed exact template/order/core binding over all
26 reference structures and all corresponding zeta levels locally.

Local CPU/float64 gate: **50 passed, 2 real-MACE tests deselected**, 15.00 s.
Covered stopping, no-cancellation absolute energy changes, strict threshold,
restart, no extra evaluations, partial epochs, failure preservation, explicit
gradient-check skip policy, cross-zeta cache reuse and concurrent single writer.

Scientific-source SHA256:
`1d249956d1e90296fe83aa697453c8b849d8a26bcceafbfc7bd01a909b52fee5`.
Uploaded source and all experiment inputs/drivers match the frozen manifest.
The ignored native `InstallScripts.json` was also copied because the current
scientific hash includes this inactive build metadata. Native compilation
uses a separate artifact build directory and never uses the Mac build tree.

The submission script now respects Slurm's two-second interval and resumes
partial submission records without duplicating jobs. SZP's attempted n005
submission was rejected by its enforced eight-hour maximum, creating no
job. The A800 update uses n004 for the 72-hour SZP jobs, and n005 only
for the four-hour preparation job. DZP/TZP retain their A100 allocations.

Remote source export:
`/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021`.
This is a new source copy, not a new Git worktree. Previous L-SZP snapshots,
training directories and checkpoints are retained.
