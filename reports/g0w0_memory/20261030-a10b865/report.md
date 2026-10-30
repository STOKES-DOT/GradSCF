# Molecular G0W0 optimization: measured snapshot and recovery status

This report distinguishes completed measurements from final acceptance, which
is pending restoration of the original SSH route. The numerical baseline was
`a10b8655b99a07dfd53557deda4d6481718534a0`. The original temporary worktree
disappeared; its committed changes and the remaining recorded file mutations
have been restored to a persistent isolated worktree. See `RECOVERY_STATUS.md`.
No merge or push has been performed.

## Implemented production paths

The molecular direct-CD residue no longer needs to return one dense auxiliary
matrix per G pole. Instead of constructing

\[
S(z)=[I-\Pi(z)]^{-1}\Pi(z),\qquad s=l^T S(z)r,
\]

the projected path solves

\[
[I-\Pi(z)]x=\Pi(z)r,\qquad s=l^Tx,
\]

and immediately reduces the result. CPU molecular QP callbacks use sequential
maps to preserve runtime skipping of inactive poles; checkpointing recomputes
the dense local intermediates in reverse AD. Periodic q0, conjugated and
off-diagonal consumers retain their existing matrix contracts. Finite eta,
static subtraction, half-pole weights, six candidate seeds, AD slopes and
original residual/branch certificates are preserved.

G/W and target windows now precede AO2MO. The code reuses `gradscf.ao2mo` for
ordered Qia, Qpm and Qmp vertices. Imaginary W uses the same ordered B[m,p]
vertex on both sides, preserving raw nonsymmetric-factor semantics; the
residue keeps its independent left/right vertices. Bare exchange uses all
reference occupied orbitals regardless of correlation windows. UGW shares
each imaginary dielectric factor between its two orbital frames.

Known arrays use the shared decimal-MB budget. It includes selected residents,
promoted input copies, frequency batches, complex solve work, Qov weighted
primal/tangent scratch and caller reservations. Small GPU dense batches are
retained only when their whole working set fits. A checked matrix-free option
uses the existing shared complex GMRES. Its real factors multiply the real and
imaginary vector parts separately, avoiding a whole complex Qov cast.

`GW(mf).density_fit(auxbasis)` and `UGW(mf).density_fit(auxbasis)` prepare
independent fixed-geometry libcint 2c/3c RI factors without rerunning SCF. When
the fits differ, F-h-J is reconstructed from the actual SCF Coulomb provider.
Stale RI cache owners are cleared before rebuilding; CPU and destination-GPU
capacity are checked before preparation/transfer. No production integral,
SCF, CC or periodic numerical kernel was replaced.

## Retained performance observations

All times below are seconds, float64. These are measurements completed before
the local directory loss, not new measurements of the recovered checkout.
`RECOVERED_SUMMARY.csv` and `recovered_measurements.jsonl` preserve evidence
and sampling limits. Complete original JSONL/XML files remain on the server.

| Workload | Before | After | Observation |
|---|---:|---:|---|
| Water/def2-SVP CPU1, warm end-to-end | 3.9762 | 0.5650 | Initial projected-residue snapshot; three warm rows recovered |
| Benzene/def2-SVP CPU64, warm GW | 334.9018 | 7.6688 | One complete baseline warm; recovered candidate repeat 3 |
| Benzene/def2-SVP CPU64, warm end-to-end | 336.5679 | 9.1818 | Same six targets and full G/W spaces |
| V100 water, warm GW | about 0.0682 | about 0.0678 | No reliable GPU speed gain claimed |

Benzene used 120 Cartesian AOs and 438 auxiliary dimensions. CPU64 corresponds
to 32 physical package/core pairs, with BLAS/OpenMP1 and native threads64. The
baseline was deliberately stopped after one complete warm because of cost;
single-sample uncertainty is not reported as zero variance. Its process VmHWM
at budget stop was 69,647,646,720 bytes, possibly including the start of an
interrupted optional repeat. The candidate's complete-process ru_maxrss was
3,791,671,296 bytes. These are process high-water observations with different
windows, not per-iteration active-tensor peaks.

The CPU8 benzene pilot was stopped after 14m31s without a returned first QP
set. It is neither an accepted speed ratio nor a 6000-second timeout result.
Independent synthetic probes also showed large reductions in compiler
temporary estimates; those estimates are not substituted for measured RSS.

The response entry was explicitly pinned to auxiliary-space direct CD, with
`resolvent_expansion=False`. The small CPU default facade already had a bounded
RPA fast path. The water result must not be advertised as a universal default
facade gain. Benzene's 2079-transition space lies outside that old fast path.

## External numerical definitions

Independent PySCF initially differed by up to 2.631 meV. A shared-input probe
found matching S/h, DF Gram matrices and MF energies, locating the difference
in bare exchange: installed PySCF GWCD defaults to exact exchange while the
tested GradSCF route uses RI exchange. With explicit `vhf_df=True`, shared-input
QP disagreement was approximately 5.46e-11 eV. This configuration alignment
did not alter production equations or loosen the 1e-3 eV external gate.

Original MOLGW b831818d/3.4 was checked against 95 archived source files;
all matched. Binary SHA256 is
`bedee25b4d99844d9a59cb834a9a509dc0d276b48103aaa5a71da0ab105075bf`.
Its first attempts failed at loader/input preparation: missing libiomp5 and
absent DEF2-SVP basis files. The valid run supplied the existing Intel runtime
only to that subprocess and exported common basis DATA into its input folder.
No reference source was patched. Completed QPs agreed with GradSCF below the
registered gate. Cross-software time ratios remain unset because frequency
algorithms, Sigma-grid work and root diagnostics differ.

## AD and final acceptance

Earlier remote checks include 35 narrow passes, an earlier 74-case
periodic/resolvent gate, UGW/AD checks and full H2 nuclear response/compile
checks. The synthetic four-parameter H2 NNAO fixture includes native contracted
RI, overlap/density/orbital response, implicit SCF, nonlinear QP loss, two FD
steps, rebinding and Adam. It does not certify trained MACE weights, force
supervision or full GPU NNAO training.

A subsequent final-source run showed one failure. Static accounting shows the
weighted-factor fixture's old 60 MB budget cannot hold its now-declared
primal/tangent scratch. Its request was corrected to 100 MB, leaving every
scientific tolerance unchanged. The exact traceback and corrected numerical
rerun are still required. AST and whitespace checks passed locally; no local
numerical tests were run. Recovered bytes must also be checked against the
remote candidate before closing acceptance.

Full packed 3c preparation and complete selected-vertex caches remain bounded
required inputs; no out-of-core provider was added. Full RI nuclear/auxiliary
metric AD was not introduced or imitated by freezing dependencies. The existing
small full-integral nuclear adapter remains the supported geometry path.

Follow the four resume steps in `RECOVERY_STATUS.md`, then finalize full raw
summaries, rerun the corrected gate, and verify/release allocation 113327.
