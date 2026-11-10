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
- [ ] Fresh-SCF/implicit AD geometry optimization to3e-6Ha/bohr.
- [ ] Complete72-column Hessian with durable per-column checkpoints.
- [ ] Raw symmetry/translation checks; 66 positive modes, no frequency clipping.
- [ ] Record actual timings, memory and derivative construction.

## C — complete electronic mode derivatives
- [ ] Full TDDFT at the accepted minimum; isolate S1 and fix signed phase anchor.
- [ ] All66 +/-mode SCF/fullTDDFT points with cross-geometry root/phase tracking.
- [ ] Two-step FD verification of important HR/HT mode union; no frozen X/Y.
- [ ] Complete provenance-checked derivative checkpoint and capability boundaries.

## D — spectrum and delivery
- [ ] Unshifted0K VG FC/HT/interference/total with optical sum rules.
- [ ] Plot physical components on common scale, mode HR factors and8vertical sticks.
- [ ] Save CSV/NPZ/PNG/PDF plus report, exact commands and limitations.
- [ ] Verify result status, preserve original workspace and release allocation.

Model is explicitly same-curvature VG (J=I), not fullAH/Duschinsky or thermal.
The initial TD facade has eigenvalue-only AD; HT uses complete displaced solves.
Initial GPUauto/no-cache and production explicit-none follow the same GPU
no-ERI-cache planner contract; production enables the verified stable program.
Do not claim a complete spectrum until all upstream scientific gates pass.
