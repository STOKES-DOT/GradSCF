# Anthracene TDDFT FC/HT — fixed execution plan

Base5b7f1d83; branch examples/anthracene-ht-20261110 in existing isolated bd0a
worktree. Preserve main checkout and prior naphthalene jobs/artifacts. Resource:
job119234, oneNV5090 +64logicalCPU +256GiB, 48h upper limit, serial stages.
User confirmed B3LYP/def2-SVP/grid3 gas-phase settings on2026-11-10.

## A — input and mathematical validation
- [x] Reuse source-identified anthracene geometry; confirm actual260AO/47occ/213vir.
- [x] Preliminary SCF/fullTDDFT converged; retain unoptimized label and costs.
- [x] Factorized VG FC/HT RED14 failures -> GREEN14 passes against independent oracles.
- [x] PySCF explicitly skipped by user; canceled reference step119234.205.
  No external validation claimed; production continues with its internal gates.

## B — ground-state geometry and curvature
- [x] Fresh-SCF/implicit AD geometry optimization:29evaluations, maxgradient1.66e-6Ha/bohr.
- [x] Complete72-column Hessian saved; finished2026-11-11 20:37+08.
- [x] Raw asymmetry6.43e-13, translation1.81e-12; all66 frequencies positive (93.9–3193.0cm^-1), no clipping.
- [ ] Record actual timings, memory and derivative construction.

## C — eight-state electronic mode derivatives (user expansion2026-11-11)
- [ ] Full TDDFT at the accepted minimum; eight targets/twelve candidates and signed phase anchors.
- [ ] All66 +/-mode shared SCF/fullTDDFT points, jointly tracking all8states; retain raw candidate data.
- [ ] Two-step FD verification of each state's topHR/topHT mode union (<=16); no frozen X/Y.
- [ ] Complete provenance-checked derivative checkpoint and capability boundaries.

## D — spectrum and delivery
- [ ] Unshifted0K VG FC/HT/interference perstate and8-state sum on one physical photon axis.
- [ ] Plot total components,2x4state bands,8x66HR factors and8target vertical sticks (guards excluded).
- [ ] Save CSV/NPZ/PNG/PDF plus report, exact commands and limitations.
- [ ] Verify result status, preserve original workspace and release allocation.

Model is explicitly same-curvature VG (J=I), not fullAH/Duschinsky or thermal.
The initial TD facade has eigenvalue-only AD; HT uses complete displaced solves.
Initial GPUauto/no-cache and production explicit-none follow the same GPU
no-ERI-cache planner contract; production enables the verified stable program.
Do not claim a complete spectrum until all upstream scientific gates pass.

Expansion record: reused all accepted ground inputs. CurrentnewGREEN36pass
(14tracking+4legacy+14factorized+4multibandplotfixtures); no actual eight-state
mode derivative is marked complete yet. OldS1 points preserved as crosschecks;
they do not contain the other signed dipoles. Same one-card allocation119234.
