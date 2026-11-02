# Corrected ethane C-C stretching: MACE, free contraction and seven conventional bases

## Status and replacement

**Complete: 504/504 accepted energy points.** This supplement replaces the
2026-10-21 Case 05 interpretation, including the five unconverged endpoints
at 2.70–2.90 angstrom and the hcore-induced energy jump. Historical inputs and
failed/candidate records are retained; no interpolation or smoothing was used.

## Methods and numerical evidence

The same 56 rigid, neutral singlet C2H6 geometries span 1.525976–8 angstrom.
All calculations use all-electron DF-RHF / def2-universal-jkfit, spherical AOs
and float64. MACE update 3351, its predicted raw coefficients, primitive
exponents and fixed core are unchanged. NNAO has 44 AOs.

The hcore guess can converge to a higher internally stable RHF solution.
At 2.75 angstrom in the same frozen MACE basis, PySCF hcore gives
-78.73767289499133 Eh, while MINAO gives -79.00203112890705 Eh; corrected
GradSCF gives -79.00203112890696 Eh. Internal stability alone does not identify
the global lowest RHF branch.

All NNAO geometries were rechecked with hcore/neutral-population guesses.
Free contraction was reoptimized at 32 distances (2.70–8 angstrom), using two
coefficient starts; 24 shorter-distance endpoints were rechecked and reused.
The accepted Euclidean-normalized coefficient-gradient criterion remains
1e-5. The current implicit SCF gradient is used. Not every rejected/local
optimizer candidate converged; endpoint acceptance and all histories are explicit.

Seven conventional bases contribute 392 accepted geometry–basis cases:
STO-3G, STO-6G, custom STO-6G+C(d)/H(p), 3-21G, 6-31G(d,p), cc-pVDZ,
and cc-pVTZ. Multiple SCF guesses were considered; 392 is not an SCF-call count.
The custom polarization exponents are C(d)=0.8 and H(p)=1.1 bohr^-2.

Independent CPU PySCF 2.9.0 recomputation passed all 112 frozen NNAO basis
cases: maximum absolute difference 1.137e-12 Eh (acceptance tolerance 1e-8 Eh).
One MINAO candidate failed numerically; other guesses resolved the same case.
This validates fixed-basis energies, not independent contraction optimization
or agreement of PySCF and GradSCF gradients. Native runs used local macOS
arm64 / JAX 0.8.1 CPU. No full four-index ERI cache is included.

## Data and figures

- [All 504 values](artifacts/nnao-ethane-dissociation-benchmark-20261022/curves.csv)
  and [AO/energy summary](artifacts/nnao-ethane-dissociation-benchmark-20261022/summary.csv).
- [Relative curves](artifacts/nnao-ethane-dissociation-benchmark-20261022/ethane-cc-relative-repaired.pdf),
  [absolute curves](artifacts/nnao-ethane-dissociation-benchmark-20261022/ethane-cc-absolute-repaired.pdf),
  [critical interval and gradients](artifacts/nnao-ethane-dissociation-benchmark-20261022/ethane-cc-critical-repaired.pdf).
  PNG previews are beside each PDF.
- [Corrected carbon radial functions](artifacts/nnao-ethane-dissociation-benchmark-20261022/carbon-radial-functions-repaired.pdf),
  [radial response](artifacts/nnao-ethane-dissociation-benchmark-20261022/carbon-radial-response-repaired.pdf),
  and [224 radial-function metrics](artifacts/nnao-ethane-dissociation-benchmark-20261022/radial-metrics-repaired.csv).
- `structures/`: 56 standard XYZ files with exact coordinate float roundtrips,
  Angstrom units, charge=0 and 2S=0. [Structure index](structure-index.json)
  maps each point to its source, distance and final basis.
- `artifacts/nnao-ethane-dissociation-repair-20261022/final-bases/`: per-point
  `frozen-bases.jsonl`, `best-coefficients.npz`, geometry and endpoint checks.
  Raw frozen-basis `structure_id` fields can retain historical torsion labels;
  use the point index, distance and exact coordinates in the structure index.
- [Flat PySCF comparison](tables/pyscf-comparison.csv),
  [all PySCF initial guesses, including failures](tables/pyscf-initial-guesses.csv),
  [five corrected endpoints](tables/critical-points.csv), and
  [full independent checks](artifacts/nnao-ethane-dissociation-repair-20261022/independent-validation.json).
- Full repair drivers, optimizer histories, 56 archived-endpoint audits and
  source hashes are retained under `artifacts/`; primitive integral, compiled
  and rendering caches are omitted. Raw absolute paths are source provenance,
  not portable replay commands. Numerical replay requires the recorded GradSCF environment.

## Units and interpretation

Absolute energies are Eh; relative energies are kcal/mol using
627.5094740631 kcal mol^-1 / Eh. Every relative curve uses its own E(R0), with
R0=1.5259763767561005 angstrom. Radial coordinates are bohr.

This is a restricted closed-shell RHF rigid-stretch stress test, **not physical
homolytic dissociation or a bond dissociation energy**. Finite-budget free
contraction is not a certified global minimum, and cc-pVTZ is a same-method
basis reference, not an exact solution. Paper-specific figures were not regenerated.
