# Nine-family NNAO training campaign

Submitted on 2026-10-13 (Asia/Shanghai). This launch supersedes the paused state
recorded in the immutable `inputs/manifest.json` preparation snapshot.

Remote checkout: `/home/user_yuanjiao/private/homefile/GradSCF_main_group1000_sp6_d3_20261013`.
Source-file hashes and Linux native-library hash are recorded in
`sync-manifest.json` and `validation-submit.json`.

## Scientific definition

- 1000 neutral singlet structures, equal weights, all-electron RHF, at most
  30 atoms including H. 25 observed main-group elements within periods 1–4.
- Original BSE 6-31G v1 inner s/p contractions: 6 primitives per radial function.
  Ga–Kr inner 3d: original 3-primitive contraction only, no separate d1 core.
- Nine SZP/DZP/TZP × S/M/L families. Per-family pool calibration uses 26 pilot
  references, up to 20 exponent steps and 300 free-contraction steps × 3 starts.
- Fresh MACE seed 0; fresh Adam, learning rate 1e-4; 1000 stage-C epochs.
  Exponents remain fixed during stage C. No held-out set.
- Primitive integrals use mmap storage. DF auxiliary basis is
  `def2-universal-jkfit`; no full four-index ERI is stored.
- Implicit SCF backward: tolerance 1e-8; SCF energy/density/gradient thresholds
  1e-12 Eh / 1e-10 / 1e-9. CaO rescue is followed by raw-Fock polishing.

## Scheduler graph

GPU validation job **96737** must succeed before any new campaign job starts.
It runs focused regression tests, then one real GPU update for all nine families
on the 26 pilot references. Production workflows retain their own numerical gates.

| Lane | S | M | L | GPU |
|---|---:|---:|---:|---|
| SZP | 96738 | 96739 | 96740 | A800, n004 |
| DZP | 96741 | 96742 | 96743 | A100 80GB |
| TZP | 96744 | 96745 | 96746 | A100 80GB |

Each lane is serial with `afterok` dependencies. SZP-S also waits for the retained
original job **96510**. Every production job requests 32 CPUs, 256 GiB host memory,
one GPU, and 72 hours. Maximum campaign concurrency is three GPUs / 96 CPUs.
72 hours is an allocation limit, not a completion estimate; no automatic
continuation job is installed. Failed dependencies do not bypass validation.

## Submission evidence

At 18:01:34 on 2026-10-13, validation 96737 was RUNNING on n001 and all nine
production jobs were PENDING (Dependency). Original 96510 remained RUNNING on
n004. Local final-core checks: 74 passed; the remote validation was still in
progress. Submission is not evidence of completed calibration or training.

Actual IDs and dependency lists: `submitted-jobs.json`.

## Kr2 stability repair, 2026-10-13

Validation 96737 failed on the 26th S-SZ pilot structure, Kr2, before any
production training started. SCF was converged; the bounded Davidson search
did not resolve all requested curvature roots at 1e-7. In diagnostic job 96759,
the GPU residuals were 1.49e-7, 8.26e-9, and 2.88e-7; dense orbital-Hessian
diagonalization confirmed positive lowest eigenvalues (3.44378, 3.44482, 3.44482).

The primary 40-vector / 100-iteration search is unchanged. Only unresolved
checks retry once with 80 vectors / 300 iterations and the same tolerance.
The final eigen residual and retry count are reported. Unresolved checks and
negative curvature remain rejected; the implicit SCF equation is unchanged.

Local CPU regression: 14 tests passed, including the reproduced Kr2 failure,
coefficient-gradient directional finite differences, CaO rescue, orbital
curvature and saddle escape. A100 job 96769 passed Kr2 and 11 other tests;
two CPU-only native-integral tests were incorrectly run with a CUDA default
and failed with a missing CPU FFI handler on CUDA. The revalidation script now
runs those native tests on CPU and the fixed-cache Kr2 regression on GPU.

New full validation **96771** supersedes 96737. All nine original production
IDs are retained with their dependencies updated to 96771; they returned to
PENDING (Dependency). Original training 96510 and its checkout remain untouched.
Full nine-family verification is still running, not yet claimed successful.
See `kr2-revalidation-submit.json`, `sync-manifest-kr2-fixed.json`, and
`submitted-jobs-before-kr2-fix.json` for provenance.

## Ge2H6 implicit-response repair and resume, 2026-10-13

Validation 96771 completed all three SZ families, then failed on Ge2H6 after
21/26 structures in the first S-DZ batch. The SCF/stability gates passed, but
the implicit adjoint true residual was 9.7058e-6 against a 6.4610e-6 threshold.
The underlying contraction projection had finite derivatives. Differentiating
individual nearly degenerate same-occupation orbitals introduced cancellation
error into the density response.

The NNAO DF-RHF fixed point now differentiates the occupied projector directly.
Its custom JVP solves the commutator response with occupied/virtual block
constraints; custom_linear_solve retains higher derivatives without taking
derivatives of the arbitrary eigenvector gauge. The forward SCF, tolerance
(1e-8), energy objective, basis, and optimizer are unchanged. A nonzero
occupied-virtual gap remains required.

Evidence:

- Original Ge2H6 regression failed locally before the fix.
- 58 local CPU regression tests passed, including coefficient finite
  differences, exact/near-degenerate projector JVP/VJP/HVP checks, Kr2, CaO,
  GMRES, saddle escape, and real MACE training/resume.
- Final CPU adjoint residual: 1.11195e-11; all 144 coefficient-gradient entries
  finite. A100 diagnostic job 96806 independently gave a 2.15100e-13 residual
  and 144/144 finite entries. GPU finite-difference regression remains part
  of the queued full resume gate.
- Both last/best checkpoint migration paths passed a byte-preservation smoke
  check for parameters, gradients, and optimizer arrays; originals are kept.

Resume validation **96809** supersedes failed 96771. It first retries S-DZ,
reuses the existing integral caches, then replays the three accepted SZ
epoch-1 checkpoints under the new source before checking the remaining
families. Only the source fingerprint is migrated; full identity comparison,
energy/gradient replay, and directional finite differences must pass. S-DZ
had no accepted epoch, so its interrupted batch is reevaluated in full.

All nine production job IDs remain unchanged and now depend on successful
96809. At the last check it was PENDING (Resources), requesting one A800,
32 CPUs, 256 GiB, and four hours. Production SZ/DZ/TZ resource requests and
original job 96510 remain unchanged. No completed nine-family validation or
1000-system training epoch is claimed.

## A100 memory request update, 2026-10-13

Per user instruction, all six DZP/TZP A100 production scripts and their
generator now request 512 GiB host memory, with 32 CPUs and one A100 80GB
unchanged. Pending jobs 96742, 96743, 96745 and 96746 were updated in Slurm
and verified to show `MinMemoryNode=512G`. Failed jobs 96741 and 96744 cannot
be edited as running/pending allocations; their restart scripts use 512 GiB.
They have not been resubmitted by this resource-only change.

SZP/A800 requests remain 256 GiB. Node n001 reports 1,031,205 MiB schedulable
RAM (about 1007 GiB), so two 512-GiB jobs cannot share that node simultaneously.
The update does not resolve the failed CPU calibration dependencies.
Scheduler before/after evidence: `memory-512GiB-update.json`.
