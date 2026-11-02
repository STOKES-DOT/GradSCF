# Ethane: one additional carbon valence-p radial channel

Compare the existing free-contraction NNAO `[C:2s1p1d; H:1s1p]` (44 AOs)
with `[C:2s2p1d; H:1s1p]` (50 AOs). Only C valence-p contraction multiplicity
changes, from one to two; each p radial column is shared by all three m components.
This is a partial valence expansion, not a complete double-zeta basis.

Use the identical 25 rigid ethane geometries, 0–120 degrees in 5-degree steps,
from `../nnao-ethane-torsion-20261018/structures.json`. Evaluate 0 and 60 degrees
first, then the remaining points. Do not relax nuclear geometry.

All primitive exponents and the 6-31G fixed core basis remain unchanged. The
152-function mixed primitive pool is identical, so reuse its checked numeric
S/h/packed-DF caches. No four-index ERI is created. The original 44-AO space
must be reproduced column for column by the first columns of the 50-AO map.

Reuse `DirectBasis`, `radial_reference_coefficients`, `FreeContractionRHF` and
the existing implicit SCF. DF-RHF / def2-universal-jkfit; float64; energy
tolerance 1e-12 Eh, density 1e-10, orbital gradient 1e-9, implicit tolerance 1e-8.
All electrons remain active. The original production basis/model APIs and
the running MACE training job are unchanged.

Optimize all active contraction coefficients from two starts: an embedding
of each previously optimized 44-AO basis, and an embedding of the reference
coefficients. Additional p columns use the existing Gaussian-overlap-metric
orthogonal initialization. L-BFGS-B, analytic gradients, maxiter 1500,
maxfun 4500, maxcor 20, maxls 30, ftol 1e-15, gtol 1e-7; report full-gradient
norm <= 1e-5 separately from SciPy termination status. No finite differences.

Check nested-space energy ordering, start dependence and SCF stability.
Export each best basis and independently verify its energy with GPU4PySCF.
Two agreeing starts do not establish a global coefficient minimum. Lower
absolute energy does not guarantee a more accurate relative torsional curve.

Add this **free-contraction** curve to the prior seven curves, in both relative
and absolute energy plots. The 44-AO MACE checkpoint stays at update 3351;
no expanded-head MACE prediction or retraining is included in this diagnostic.

Run: `sbatch artifacts/nnao-ethane-p2-20261018/run.slurm`.
Resources: one V100, 32 CPU, 128 GiB host RAM, two-hour limit.

## Symmetry completion

Phase 1 (job 100401) completed independent two-start optimizations at all 15
angles from 0 through 70 degrees. All completed results and the interrupted
75-degree attempt were archived before stopping repeated equivalent-angle
optimization. Phase 2 (job 100454) retained these endpoints and mapped the
remaining 10 angles by exact reflected atom permutations, phi -> 120-phi.

Each mapped angle received a new converged/stable SCF energy and full
coefficient-gradient calculation. Acceptance required norm <= 1e-5, mirrored
energy difference <= 1e-8 Eh and gradient maximum difference <= 2e-7. All
25 final frozen bases were then independently checked with GPU4PySCF. No
curve interpolation or synthetic energy points were used. Independent
optimization and symmetry-transfer counts are recorded separately.

The original MACE training job 99005 was not changed. Continuation command:
`sbatch --dependency=afterany:100401 artifacts/nnao-ethane-p2-20261018/complete.slurm`.
