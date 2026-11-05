# Follow-up: measured RKS cost and next production changes

The accepted cached-AO GradSCF RKS/PBE/RI-J warm median is **535.544680 s**,
versus **211.686044 s** for the certified PySCF reference: a 323.858636 s gap.
These are complete-workflow medians, not the profiles below. Atomic guesses,
stopping workflows and RI representations differ; this is not an implementation-only speed ratio.

| Existing complete-workflow record | Cold (s) | Warm median (s) | Acceptance |
|---|---:|---:|---|
| GradSCF AO reuse, RI-J | 577.914945 | 535.544680 | 4 calls; 23 cycles each |
| PySCF certified continuation, RI-J | 212.216112 | 211.686044 | 4 calls |
| Original `40a569bf` conventional direct | 795.437445 | 754.542832 | 4 calls; 22 cycles each; candidate pair unfinished |

Sources: [candidate RI](candidate_rks_pbe_ri.json), [certified reference](reference_certified.json),
[original direct](baseline_rks_pbe_direct.json). Direct and RI are separate numerical tasks.

## Scope

Production remains **`1a575d18`**. This round added diagnostic harnesses and
prototype array expressions only; **no new production benefit is claimed**.
The independent reference uses the installed PySCF 2.13.0, without GradSCF
orbitals or integrals. [followup_manifest.json](followup_manifest.json) records
the worktree, source/probe identities, resource limits and unsupported branch.
Runs used CPU float64, 64 logical CPUs on n006/Slurm 115927: C40H56,
Cartesian def2-SVP/PBE/grid3, 880 AO, 3112 auxiliary functions, 1,113,968 points.
Both shared probes loaded the same converged PySCF D, SHA256
`470beec5d2cf02363f7bc9410babe366f2220671db1525e13d47f10683914f71`.
[followup_summary.json](followup_summary.json) retains raw hashes;
[followup_before_after.csv](followup_before_after.csv) labels every row's scope, units and samples.
Inclusive CPU spans contain instrumentation overhead. Nested spans cannot be
added; XLA memory estimates are not measured RSS.

## GradSCF fixed-five profile

The **355.239301 s** facade is a deliberately unconverged snapshot:
`max_cycle=5`, `conv_tol=0`, `scientific_pass=false`. It is not an end-to-end
speed result or a substitute for a converged 23-cycle measurement.

| Inclusive diagnostic span | Seconds | Included children / qualification |
|---|---:|---|
| Input preparation | 174.440168 | Includes integral/grid preparation |
| RI factors | 106.439757 | Native 2c/3c 26.496248; host whitening 78.657629 |
| Factor unpacking | 18.430679 | Separate recorded span |
| Solver entry | 180.731661 | Includes AO retention 101.124147 and initial Fock 14.180031 |

The cold process recorded **521 `backend_compile_and_load` calls**,
8.061930 s cumulative / 8.061649 s self time. This is not a warm fresh-object
compilation count, nor evidence that AD has no overhead. No AD-overhead
isolation was performed. RKS eager history is skipped because the capture may
belong to an atomic initializer, not the molecular JIT loop.

## Independent PySCF profile

The instrumented logical SCF was **accepted in 212.927215 s**, with 28 main
cycles and two extra checks across two kernel entries. Strict thresholds and
bounded continuation match the certified variant; this is not a clean repeat.

| Synchronous inclusive stage | Calls | Total seconds |
|---|---:|---:|
| `mf.get_veff` | 32 | 203.924066 |
| `nr_rks` | 32 | 181.849630 |
| RI-J `with_df.get_jk` | 32 | 18.693917 |
| Grid build | 1 | 3.333052 |
| AO evaluation, nested | 544 | 26.776632 |
| Point XC, nested | 544 | 15.290688 |
| Sparse density products, nested | 544 | 50.730707 |
| Sparse Vxc products, nested | 544 | 67.393322 |
| Sparse AO scaling, nested | 544 | 4.273308 |

`nr_rks` accounts for **85.4%** of this diagnostic's elapsed time. Its children
are already included, not additional costs. The metadata observer never calls
`get_naoaux()` or builds missing factors. This supports integration/contraction
priorities, not a causal partition of the cross-engine gap.

## Shared-input RI-J and host preparation

Production PBE correctly skips K. The avoidable candidate is the full-factor
preparation before that J-only consumer, rather than a spurious K evaluation.
Native metric/3c preparation took 25.198864 s and Cholesky factorization
0.438193 s. BLAS1 host-whitening samples were 78.406058, 78.454096 and
78.597825 s: **median 78.454096 s**. This includes finite checks and the host
solve, excluding comparison. It is not the instrumented 78.657629 s span.
**BLAS8 is unsupported** because `threadpoolctl` is absent: no sample was
invented and no dependency was installed.

| Same-D J kernel | Warm median, 3 calls (s) | XLA temporary estimate (bytes) | Max J difference |
|---|---:|---:|---:|
| Full whitened L | 0.313333 | 6,195,200 | 0 |
| Packed whitened L | 0.723804 | 9,666,590,120 | 6.821e-13 |
| Raw C + Cholesky-metric solve | 0.788980 | 9,744,066,472 | 1.893e-11 |

Values pass `atol=rtol=1e-10`. Packing reduced input storage but was slower
and introduced large compiler workspace: **blind packed replacement is not a
retained optimization**. Raw C could avoid whitening and full-L unpacking
(18.457275 s in this probe), but each J is slower; bounded layout, full SCF
and metric/basis/geometry AD remain unvalidated. Tiny packed-factor/D JVP/VJP
does not validate raw-metric AD. Only cumulative RSS was recorded, so no
per-variant actual peak comparison is available.

Source: [whitening:46-57](../../../src/gradscf/integrals/molecular/density_fitting.py#L46-L57),
[factors:190-212](../../../src/gradscf/integrals/molecular/density_fitting.py#L190-L212),
[unpacking:228-236](../../../src/gradscf/integrals/molecular/density_fitting.py#L228-L236),
and [prototype expressions:75-84](../../../benchmarks/probes/scf_100/probe_ri_j.py#L75-L84).

## Shared real 4096-point XC tile

| Rearrangement | Old warm median (ms) | New warm median (ms) | Reduction |
|---|---:|---:|---:|
| Shared density projection | 20.294194 | 14.071486 | 30.66% |
| Weighted-AO Vxc | 33.602267 | 26.978726 | 19.71% |

Each median has ten synchronized calls after separate compilation. Reusing
`C = AO @ D.T` reduces optimized density projection dots **2 -> 1**.
Weighted-AO construction reduces **880x880 Gram products 2 -> 1**. Total
syntactic HLO dots fall 3 -> 2, but the remaining 4096x880 derivative-field
blend is not another O(N_AO^2 N_grid) Gram product. Counts are compiled
expressions, not runtime invocations.
Tiny independent nonsymmetric D/weight/all-array value, JVP and VJP checks
passed. Real-tile max rho/gradient difference is 9.770e-15; Vxc is 2.220e-16.
No complete geometry or SCF gain follows from these microkernels.
Production: [features.py:224-243](../../../src/gradscf/tools/features.py#L224-L243),
[rks.py:429-456](../../../src/gradscf/scf/rks.py#L429-L456); candidates:
[probe_xc_algebra.py:62-72](../../../benchmarks/probes/scf_100/probe_xc_algebra.py#L62-L72).

## Priorities and reproduction

1. Implement the shared density/Vxc rearrangements as one small production
   patch, without new approximations; require physical geometry/NNAO/neural
   regressions and matched complete-workflow acceptance before claiming gains.
2. Prototype pure-J raw-metric products with bounded layouts, avoiding the
   observed XLA copies. Current packed/raw variants are not production replacements.
3. Retest scoped host BLAS threading with a supported control and recorded
   environment; BLAS8 has not run.
4. Probe AO/value/derivative parity on identical coordinates and layouts.
   GradSCF first retention 101.124 s and PySCF cumulative AO 26.777 s are not
   matched inputs/algorithms and have no valid speed ratio. Preserve NNAO
   coefficient/exponent/mixed derivatives. [Materializer:210](../../../src/gradscf/integrals/grids/blocks.py#L210).

Native parallel efficiency is **source_only**: no scaling/utilization evidence
was collected. Configured thread counts do not establish the cause of the gap.
Payload examples below require the existing CPU64 Slurm/external guard and
candidate-source PYTHONPATH in [controller.sh](raw/followup-v1/controller.sh):

```bash
CFG=reports/scf_100/20261105-40a569bf/plan-v2/configs
python benchmarks/probes/scf_100/profile_followup.py \
  "$CFG/beta-carotene-def2-svp-rks-pbe-df-gradscf.json" NEW_FIXED5_OUTPUT \
  --mode diagnostic --fixed-cycles 5 --grid-block-size 0 --budget-seconds 1200

# Generate the shared density when the archived raw payload is unavailable.
python benchmarks/probes/scf_100/profile_pyscf.py \
  "$CFG/beta-carotene-def2-svp-rks-pbe-df-pyscf.json" NEW_REFERENCE_OUTPUT
python benchmarks/probes/scf_100/probe_xc_algebra.py \
  "$CFG/beta-carotene-def2-svp-rks-pbe-df-gradscf.json" \
  NEW_REFERENCE_OUTPUT/density.npy NEW_TILE_OUTPUT
```

All four guards returned 0; this certifies execution, not unconverged or
prototype scientific acceptance. The original queue resumed at **2026-11-05
16:54:01 +08:00**. The 30-configuration matrix per engine remains unfinished;
no production, threshold or original benchmark record changed in this round.
