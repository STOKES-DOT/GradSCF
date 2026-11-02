# Dimer interactions: independent monomer references

**Corrected on 2026-10-27.** The main figures now use independent monomer
energies that are constant through each rigid separation scan. Original raw
distance-point calculations are unchanged. The previous dimer-conditioned
reference figures/tables are preserved in
[`archive/conditional-reference-v1/`](archive/conditional-reference-v1/README.md).

- [Current two-group figure gallery](figures/index.html)
- [No CP: 12-page PDF](figures/dimers-no-cp-with-structures.pdf)
- [Model + frozen-parameter CP correction: 12-page PDF](figures/dimers-cp-with-structures.pdf)
- [Exact main-curve values](combined/curves.csv)
- [Constant isolated references](isolated-reference-map.json)

## Reference and output policies

The checkpoint remains historical **update 3351**. Both MACE curves use the
same independently predicted isolated monomers. Fixed output freezes **the
scanned dimer's** coefficients at distance scale 1.00; adaptive output uses
the dimer prediction at each geometry. Network weights and primitive
exponents are fixed. This is a dimer-output response comparison with a common
model reference, not a separately redefined frozen-basis monomer model.

For every basis, the no-CP curve is

\[
\Delta E_{\rm iso}(R)=E_{AB}(R)-E_A^{\rm iso}-E_B^{\rm iso}.
\]

The isolated monomer geometry, MACE prediction/basis and its independent DF
auxiliary space are evaluated once and held unchanged across the scan. Existing
reference energies were reused where available. The 24 missing STO-6G+P
monomer energies were computed with native DF-RHF and real internal-stability
checks; those results are in `isolated-results.json`.

## CP-adjusted figure convention

The CP group uses an explicitly defined **frozen-parameter CP estimate**:

\[
\Delta E_{\rm adj}(R)=\Delta E_{\rm iso}(R)+\delta_{\rm CP}^{\rm frozen}(R),
\qquad
\delta_{\rm CP}^{\rm frozen}=E_A^{\rm own}+E_B^{\rm own}
-E_A^{\rm ghost}-E_B^{\rm ghost}.
\]

The CP increment comes directly from the original five calculations, using
the selected fixed/adaptive dimer coefficients unchanged in all five energies
and one common dimer auxiliary metric. The main curve **retains** the
reference/basis-response term relative to that conditional diagnostic:
`CP-adjusted = conditional CP + reference_response`.

For an environment-dependent basis, this composite estimate is not identical
to the previous bare ghost-subtraction diagnostic. Independent monomer and
common-dimer auxiliary approximations can also differ slightly. Both effects
are explicit in `reference_response_kcal_mol`; no constant vertical shifts,
short-range clipping, forced asymptotic zeros or energy refitting are applied.
A fixed-dimer-output curve can retain a far-separation basis-relaxation offset
relative to the shared isolated model reference.

## Coverage and reproducibility

All 12 dimers and 108 geometries are unchanged. The figures show both MACE
policies, STO-6G, STO-6G+P, 6-31G(d,p), def2-SVP, cc-pVDZ and cc-pVTZ, with
the original reference curves. DF-RHF / def2-universal-jkfit, spherical AOs,
float64 and the original convergence settings are retained. Correlated-source
reference curves include method differences as well as basis differences.

Verification checks all 1728 calculated curve values, 216 displayed reference
values, 96 constant reference groups, shared MACE references and both 12-page
PDFs. The HBr–NH3 scale-0.80 no-CP adaptive value is **+1.079335 kcal/mol**;
its frozen-parameter CP-adjusted value is **+5.145348 kcal/mol**.

`combined/results.json`, `run/results.json` and `inputs/cases.json` remain the
unchanged original records. `assemble.py` rebuilds the corrected CSV;
`plot_controls.py` redraws figures; `verify_results.py` checks current exports.
The original numerical drivers remain in `drivers/`; correction helpers and
the monomer calculation are in `reference-correction/`.

The historical reference diagnosis remains in `diagnostics/`. No MACE model,
optimizer, training state, publication text or prior case archive was modified.
