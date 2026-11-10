# HFKS-03c — device ownership and reuse policy

Status: scoped ownership/reuse validation passed, including final CPU/CUDA
regressions and four converged molecular MGGA cache/stream comparisons.
No commit. The overall fixed plan still has representative/count evidence
being completed separately by the coordinator.

## Evidence

The coordinator observed job 116774 phase 15:

- tests/test_scf_grid_device_ownership.py: 9 failed / 9 passed in 3.45 s.
  Failures exposed host-native work incorrectly charged to GPU capacity,
  missing explicit single-consumption handling, and missing ownership
  forwarding at restricted/unrestricted/generalized SCF boundaries.
- tests/test_mgga_grid_reuse.py: four failures in 31.18 s, covering real
  r2SCAN and M06-2X with restricted/unrestricted multi-block AO consumers.

The separate metadata-only assembly regression was observed in phase 14:
3 failed / 4 deselected in 1.52 s. The budget forecast now uses the existing
NativeDirectBasis(plan, None) placeholder. The real execution provider still
binds current parameters.

## Reservation ownership

The private _scf_operator_reservation returns (device_bytes, host_bytes).
The existing scf_operator_workspace_bytes retains its scalar total for callers
that need a method-wide estimate. Both paths use the same underlying logic.

RI auxiliary vectors and full QNN exchange products belong to the JAX consumer
device. Unrestricted RI retains its second simultaneous QNN product. Canonical
native direct work belongs to the CPU, including GPU callbacks, and still
reserves three overlapping calls. Unknown and projected providers return None,
preserving conservative optional-cache fallback.

Let M be the requested method budget after the existing 20% reserve, A the
detected device capacity after the same reserve, H the host-native reservation,
and R the device resident arrays including raw AO-grid leaves. GPU cache
admission uses:

    available = min(M - H, A) - R

Host storage remains charged against the total method budget. On CPU both
categories share one pool, so admission keeps:

    available = min(M, A) - H - R

The cache-output, copy, and bounded scratch reservations remain the existing
ones. This is an ownership correction, not a relaxation of total method memory.

## Reuse and functional consumers

prepare_scf_grid accepts an internal expected_passes hint. Explicit 0 or 1
retains streamed AO production; None preserves existing admission behavior.
No max_cycle value is interpreted as the actual number of Fock evaluations.

RKS and UKS now admit the already supported tau-MGGA path to the same
four-channel AO cache used by GGA. Existing density/tau/Vxc code and Laplacian
guards are unchanged. Generalized SCF retains its existing functional support.

## Verified GREEN and measured reuse

Direct inspection of the synchronized XML confirms:

- [policy-green.xml](final_raw/policy-green.xml): **62 passed**, zero failures,
  errors or skips, 49.612 s. Includes all 18 ownership tests, four MGGA reuse
  tests, all seven integral-budget tests, and the existing grid/cache suites.
- [final-cpu.xml](final_raw/final-cpu.xml): **94 passed**, zero skips, 215.832 s;
  repeats policy/MGGA/budget cases alongside six-facade, preparation and training.
- [final-gpu.xml](final_raw/final-gpu.xml): **28 passed**, zero skips, 283.817 s;
  native GPU bridge, streamed external XC and an actual GPU NNAO update.
- [mgga-real.xml](final_raw/mgga-real.xml), phase 23a: **4 passed**, zero skips,
  69.082 s. These are fully converged molecular calculations with a real
  adaptive AOGrid, actual _CachedAOGrid, and explicit streaming comparison.

The converged MGGA fixture uses Cartesian def2-SVP/grid3, 128 decimal MB,
native direct J/K with two workers and cutoff 1e-12. Water RKS uses 25 AOs/
33,698 points and OH UKS 20 AOs/23,890 points. Cached/streamed cycles match:

| Functional | Water RKS cycles | OH UKS cycles |
|---|---:|---:|
| r2SCAN | 12 / 12 | 23 / 23 |
| M06-2X | 13 / 13 | 23 / 23 |

[mgga-real.log](final_raw/mgga-real.log) records maximum absolute energy
difference 4.264e-14 Ha, density difference 1.948e-13, Fock difference
8.932e-14 Ha, and orbital residual below 2.202e-9 Ha across these four cases.
The test gates remain 2e-8 Ha energy and 2e-7 density/Fock, rtol=0.

[The fixed-density large-grid measurement](final_raw/mgga-reuse/results.json)
records 34.449135 s preparation. PBE streamed/cached warm medians are
12.674815/10.168400 s; r2SCAN medians are 32.447140/30.961822 s. Based on
these measured warm medians, preparation amortizes after 14 and 24 passes,
respectively. This is specific to the measured grid/device and excludes
compilation differences; max_cycle does not establish actual amortization.
The molecular tests establish correctness, not a universal MGGA SCF speedup.

Final PBE/direct beta-carotene anchors also passed four of four calls on
both CPU and RTX 5090. Their complete call timing and source lineage are
recorded in [HFKS-04ab.md](HFKS-04ab.md#final-source-and-large-anchor-check).
Test counts overlap and must not be summed as unique coverage.

## Validation boundary

An independent read-only review found no Critical or Important blocker in the
ownership arithmetic, RI reservations, three-call native allowance, conservative
fallbacks, functional conditions, or metadata-only assembly fix.

This correction applies at the SCF execution cache-admission boundary. The
earlier assembly-to-_streamed_grid fixed_workspace_bytes argument remains a
conservative combined reservation. Direct GPU-context input assembly with a
host reservation larger than VRAM needs a separate entry-path regression; it
is not established by these runtime-cache tests.

Local verification is limited to Python syntax and diff whitespace checks.
The 18 ownership/policy tests use synthetic capacity limits and real small CPU
AO values. They do not establish physical GPU allocator behavior or performance.
The MGGA fixtures supply actual repeated consumer and numerical comparisons.

The scoped GREEN gates above are complete. Remaining overall-plan evidence
is the representative campaign rerun after a harness-only guard-argument
error, and a separate diagnostic count sample. These are not substituted by
kernel timings or inferred from SCF cycle counts. Final full-call compilation
is included in solver time, not separately attributed. Production/consumer
tile sizes are unchanged; rejected contiguous-cache slicing remains absent.

## Evidence fingerprints

| Artifact | SHA256 |
|---|---|
| final_raw/policy-green.xml | `86fcea465a4181d1c394c06018f96f2a5d53fbd5502a85077ddcfac9c249555f` |
| final_raw/final-cpu.xml | `dfff341e6364932ae60d981b3d009e23382ffb76edddb5bcda844f7bffb82c80` |
| final_raw/final-gpu.xml | `98572d0454dfcdd33e42496f747f03d2fb3a2b9674127d7af0b282999a3c6767` |
| final_raw/mgga-real.xml | `fbfe1263a90c7d51655bce3f287f8722c0ae043d6a8ee2ff689bf8895bf63286` |

The five policy/consumer source hashes match the final anchor runtimes and
the current checkout. All eight retained production-file hashes are listed
in [HFKS-04ab.md](HFKS-04ab.md#final-source-and-large-anchor-check).
