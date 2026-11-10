# SCF-100 implementation and acceptance

This is the implementation follow-up to `CONVENTIONAL.md`. The frozen physical
task and acceptance gates remain in `TDD_PLAN.md`; earlier raw results are not
overwritten. Large-system results below refer to beta-carotene C40H56, 96 atoms,
Cartesian def2-SVP (880 AO), and grid level 3 for KS. CPU64 means 64 logical
CPUs on 32 physical cores with SMT, not 64 physical cores.

## Production changes

### Reuse semilocal contractions

`tools/features.py::_spin_density_and_gradient` shares the AO-density product
between rho and its gradient. `scf/rks.py` combines the rho and GGA-gradient
potential contractions before the final symmetric matrix sum. UKS and the
collinear GKS path reuse this helper. No XC definition, grid, or response
channel is removed; the nonsymmetric/complex algebra and tau/laplacian terms
remain intact. These changes introduce no approximation.

### Screen incremental direct J/K explicitly

The libcint interface accepts an optional `screen_density=S`. For fixed S,
the retained operator M(D; S) is linear in D. If u_ij is the maximum over
reference batches of the shell-pair L1 norm of S, set s_ij=max(u_ij,u_ji).
For Schwarz bounds B_ij, retain a canonical quartet if both

```
B_ij B_kl >= cutoff
8 B_ij B_kl max(s_ij,s_kl,s_ik,s_il,s_jk,s_jl) >= cutoff.
```

All symmetry copies share one decision. The rule is independent of J-only,
K-only, or combined selection. A cheap outer bound can reject an outer shell
pair before its inner loop without changing this retained set. This is a local
contribution bound for the selected reference, not a bound on the accumulated
Fock, energy, or SCF error. Positive screening is a numerical approximation;
fresh convergence certification uses the original S=None operator.

S is an explicit dynamic operand rather than hidden provider state. Density
JVP/VJP products retain primal S; its discrete subset indicator has zero
derivative away from thresholds. The default S=None preserves the original
linear response contract. With cutoff=0, the original exact geometry, basis,
and projected-coefficient paths remain in use. Existing positive-cutoff
geometry and primitive-basis AD rejections are retained. There is no new
physical stop_gradient boundary and no detached numeric cache.

RHF/RKS, UHF/UKS, and GHF/GKS use their actual incremental density as S only
for the supported native forward consumer. GHF/GKS batch all four complex
spin blocks, preserving off-diagonal exchange. RSH uses the same reference
for its separate full- and long-range operators. RI-J/direct-K remains that
explicit approximation; true RI-JK is not replaced by this path.

### Rebuild and certify converged states

The shared SCF solver now performs a full evaluation when the original gates
pass, or when density and gradient are ready but accumulated energy drift
prevents the energy gate. The fresh evaluation must still pass all three
original gates. If it does not, its E/F/aux become the next iteration's
reference. The selected historical best state's values are also rebuilt when
needed. Fixed-cycle conv_tol=0 behavior is unchanged.

UKS reuses the certified selected state's E/Exc/F instead of evaluating them
unconditionally again. Bound XC callbacks with nonzero level shift require
special ordering because they may depend on reported orbital energies; the
dedicated regression records that contract. Advance and fresh evaluation
counts are separate SCFResult fields.

## TDD and current evidence

The remote source is based on 9733a3b4 plus independently hashed patches.
Fresh native output SHA256 is
`c7c9b86695e15e356bb934d494327bd8d55df3bd465e0753c5ea399f570f3586`.
The previous unscreened library SHA256 is
`ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8`.
Tests run only in Slurm allocation 115927 on n006, CPU8/float64 for narrow
regressions. Formal large rows use CPU64 and the unchanged input hashes.

| Group | Observed result | Scope |
|---|---|---|
| Shared XC RED | 13 passed, 2 failed as intended | Existing formulas pass; structural reuse assertions fail |
| Shared XC GREEN | 15 passed, 11.417 s | Real/complex value, JVP/VJP/HVP and shared contraction |
| Screen/refresh RED | 39 failed, 4 passed | Missing screen interface and reliable rebase behavior |
| Combined core GREEN v2 | 58 passed, 24.868 s | XC 15, native screen 20, reliable SCF 23 |
| Existing core v2 | 85 passed, 2 failed | One expected test adaptation; one pre-existing CC/reference boundary |
| Pre-existing precision failure | Reproduced on parent | `test_precision_layer_report_detects_tight_scf_target`; not an introduced failure |
| Neural/custom compatibility v2 | 37 passed, 269.46 s | Three training modes, force mixed derivatives, custom callback semantics |
| Combined XC compatibility process | SIGABRT during compilation | Preserve as unresolved process-level failure; no full-group pass claim |
| Isolated cache HVP/FD node from abort | 1 passed, 7.78 s | Same node and input on the same allocation |
| Geometry/NNAO v2 | 8 passed, 92.36 s | UKS/GKS geometry, direct/RI coefficients, native log exponents, small MACE update/checkpoint |
| Bound-XC level-shift RED | 1 failed, 2.53 s | Demonstrated stale eps passed to the final callback |
| Core GREEN v3 | 46 passed, 15.11 s | SCF 23, screen 22 including private/spherical, corrected bound callback 1 |
| Existing core v3 | 89 passed, 1 deselected, 80.00 s | Only the independently reproduced parent precision failure excluded |

The historical incremental-energy toy test has been adapted, retaining its
original fixture and thresholds: after two cycles all mismatches must fail;
after a third fresh rebase only the constant-energy mismatch may converge,
returning the correct fresh energy. Requiring the deliberately drifting
incremental energy to equal fresh energy would add a new fourth gate and
prevent the authorized rebase. Original test bytes and their hashes remain in
the raw evidence directory.

The new screening oracle initially covered row-owned batches. Two additional
passing forward oracles cover single-matrix private-output execution with both
Cartesian and spherical AO layouts. The geometry/NNAO and neural groups ran
on v2; the v3 change only restores the nonzero-level-shift bound-callback
branch, covered by its RED/GREEN regression. Their original source snapshots
remain identifiable. Test collection or submission is not a passing result.

One existing HF lifecycle limitation remains: UHF inherits a builder which
eagerly supplies a grid for a later neural initializer. The ordinary HF kernel
does not evaluate XC, but the timed UHF facade includes that preparation.
`test_unrestricted_hf_reference_has_grid_for_neural_initializer` explicitly
protects the behavior. This round preserves that compatibility rather than
silently deleting the grid. RHF/GHF are grid-free; no claim is made that every
HF facade is grid-free, or that the full RHF/UHF timing difference is grid cost.

## Large-run accounting

The old matrix controller was stopped at 2026-11-05T19:37:59+08:00 before
implementation. Its partially started PySCF GHF row is an agent-requested
interruption, not a timeout or numerical failure. Completed baseline rows
remain valid only for their recorded source. Optimized rows use a new source
manifest and output prefix. No incomplete or nonconverged reference row is
used as a speedup denominator.

Large optimized rows started at 2026-11-05T20:55:04+08:00 in
`matrix-optimized-v1`, first covering direct RHF/RKS-PBE/UHF/GHF/UKS-PBE/GKS-PBE.
The source snapshot `candidate-optimized-v1` contains 1125 regular source
files, all byte-identical to the tested v3 tree after excluding bytecode.
Remaining results are pending at this checkpoint. Preparation, cold,
fresh-object warm, convergence gates, peak process RSS, and native/source
provenance must be reported together. The earlier AO-cache-only improvement
does not certify these new patches. Scientific tests do not run on the local
Mac or the remote login node.

The first optimized row completed at 2026-11-05T20:58:47+08:00:

| Conventional RHF, full facade cold call | Parent 1a575d18 | Optimized v3 |
|---|---:|---:|
| Elapsed seconds | 271.453568 | 218.254523 |
| Main cycles | 21 | 23 |
| Total energy / Ha | -1546.2032025822882 | -1546.2032025822136 |
| Occupied MO gradient / Ha | 2.26795e-8 | 3.89063e-9 |
| Sampled whole-process-group peak RSS / bytes | 3709173760 | 3802742784 |

This is a 19.5978% cold elapsed reduction with a 7.45786e-11 Ha energy
difference. Both scientific gates passed. It is one cold sample per source,
not a warm median, a kernel-only speedup, or a six-method aggregate. Peak RSS
increased by approximately 94 MB; it is not a memory reduction claim. The
parent and candidate share the same input hash, affinity, dtype and worker;
the candidate's incremental screening and refresh can change the iteration
trajectory. Original raw records are in `raw/matrix-optimized-v1/` and
`raw/implementation-v1/remote/results/matrix-v1/`.

At 21:30 +08:00 the optimized conventional PBE row had completed its cold
call (635.354611 s) and two warm calls (590.778647, 590.242156 s), all with
26 cycles, E=-1554.7025267060226 Ha and g=5.97663e-9 Ha. The fourth call was
still running. These times improve on the original 40a baseline but regress
against the cached 1a parent (587.118939 s cold, warm median 544.734456 s,
22 cycles). A complete four-repeat result is required before summarizing the
new row. No positive PBE speed claim is made for the new incremental patch.

The waiting controller alone was paused, allowing that scientific process to
finish unchanged. A single-factor source omitting only RKS's new S forwarding
is prepared; its other 1124 source/native files match v3. A separately labelled
one-cold diagnostic keeps the registered physical input and stopping gates,
but cannot satisfy the original four-call acceptance. Its helper records that
repetition-only overlay and refuses pre-existing output directories. The UHF
lazy-grid RED follows the current PBE process before this cold probe, with all
scientific steps serialized in the same allocation.
