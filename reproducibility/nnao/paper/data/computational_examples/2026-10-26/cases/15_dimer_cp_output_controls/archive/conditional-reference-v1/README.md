# Dimer interaction controls: CP × MACE output

**Energy-reference clarification (2026-10-27):** the adaptive curves use distance-dependent dimer-conditioned monomer references. Read the [interpretation and independent numerical check](diagnostics/hbr-nh3-reference-20261027/INTERPRETATION_NOTE.md) before treating them as model potential-energy curves. Original energies and figures remain unchanged.

[Two-group figure gallery](figures/index.html).
[No-CP 12-page PDF](figures/dimers-no-cp-with-structures.pdf).
[CP-corrected 12-page PDF](figures/dimers-cp-with-structures.pdf).

Historical checkpoint update 3351; DF-RHF / def2-universal-jkfit, spherical AOs,
float64. All 12 representative dimers and 108 geometries are unchanged.
Fixed MACE output uses the coefficients predicted at distance scale 1.00 for
the entire scan; adaptive output uses the archived prediction at each geometry.
Network weights and primitive exponents remain fixed in both policies.
At each geometry the selected coefficients are identical in all five CP
calculations, sharing the dimer auxiliary metric. Electronic SCF relaxes anew.

The no-CP panel uses E_AB - E_A(own) - E_B(own); the CP panel uses
E_AB - E_A(ghost) - E_B(ghost). Independent-monomer model interaction energies
are excluded from this matched comparison. Axes use dimensionless source
distance factors and kcal/mol, with equal y limits for both panels of a dimer.

Controls: STO-6G, STO-6G+P, 6-31G(d,p), def2-SVP, cc-pVDZ and cc-pVTZ.
STO-6G+P adds the most diffuse original uncontracted def2-SVP p(H)/d(heavy)
shell. Explicit assets preserve reference exponents, including Br d = 0.389.
Correlated-source reference curves also contain electronic-method differences.
The source benzene TS label denotes the T-shaped dimer, not a transition state.

There are 204 new five-energy cases and 660 reused records, including twelve
identical fixed/adaptive anchors: 864 records, 1728 calculated curve values,
plus 216 displayed reference values. Four new cases were checked first on CPU;
the continuation records its hardware per case. Native SCF/stationarity and
real internal stability were checked for all five energies of each new case.
Reused records retain their original validation scope; global HF minima and
external spin/complex stability are not certified.

- `combined/curves.csv`: exact CP/no-CP values and source fields.
- `combined/results.json`: all merged records with original diagnostics.
- `run/results.json`: unchanged new calculations, with per-case hardware/time.
- `inputs/cases.json`: full input geometries, frozen GTOs, anchor identities and
  source/checkpoint hashes. `inputs/polarization-assets.json`: STO-6G+P assets.
- `representative-structures/`: existing reference-geometry molecular renders.
- `verification.json`: exact arithmetic, anchor and plot-value checks.
- `RUN_SPEC.md`, `drivers/`, resource reports and Slurm scripts: reproduction
  evidence; numerical drivers require the recorded frozen GradSCF environment.

No finite differences or full four-index ERI were used. Original benchmark
archives, existing paper figures/scripts, manuscript text and training jobs
were not modified by this handoff. Generated figures received numerical and
layout-bound checks; a separate model visual review is not claimed here.
