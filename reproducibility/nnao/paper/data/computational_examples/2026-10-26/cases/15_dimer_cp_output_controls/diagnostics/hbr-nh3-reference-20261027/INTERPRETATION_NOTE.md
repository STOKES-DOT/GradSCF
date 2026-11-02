# Energy-reference clarification for adaptive-output dimer curves

**Historical diagnosis:** this note describes the archived `conditional-reference-v1` exports. Current main figures were corrected to constant independent monomer references; see [current definitions](../../README.md).

Confirmed 2026-10-27 after inspecting HBr–NH3 at short separation. The archived
numeric values and CP identities are reproducible; interpreting the adaptive
curves as the independent-monomer model potential requires a different
reference definition. No archived energy, figure or training parameter was
changed during this investigation.

## Two distinct quantities

The actual independent-monomer model interaction is

\[
\Delta E_{\mathrm{model}}(R)=E_{AB}[c_{AB}(R)]-E_A[c_A]-E_B[c_B],
\]

where each isolated monomer receives its own MACE prediction. The no-CP
adaptive curve displayed in Case 15 instead uses

\[
\Delta E_{\mathrm{conditional}}(R)=E_{AB}[c_{AB}(R)]
-E_A[c_{AB,A}(R)]-E_B[c_{AB,B}(R)].
\]

These fragment bases retain the dimer's environmental prediction even after
removing its partner. Their own-fragment energies change with separation.
The difference is the explicit reference/basis-response term

\[
\Delta_{\mathrm{ref}}(R)=E_A[c_{AB,A}(R)]+E_B[c_{AB,B}(R)]-E_A[c_A]-E_B[c_B],
\qquad
\Delta E_{\mathrm{model}}=\Delta E_{\mathrm{conditional}}+\Delta_{\mathrm{ref}}.
\]

The exported difference also includes the very small difference between the
common-dimer and independent-monomer auxiliary approximations/geometries.
Internal fragment distances vary by at most 9.4e-10 Angstrom through this
HBr–NH3 scan; ordinary fixed-output own-energy variation is tiny relative to
the observed adaptive reference shift.

## Concrete numbers (kcal/mol)

| Distance scale | Model, independent monomers | Displayed conditional no-CP | Reference shift | Displayed conditional CP |
|---|---:|---:|---:|---:|
| 0.80 | +1.079335 | -12.850345 | +13.929680 | -8.784331 |
| 1.00 | -7.844951 | -9.040979 | +1.196028 | -7.722164 |

At scale 0.80, adaptive minus fixed output changes the dimer total energy by
-0.413097 kcal/mol, but changes the summed own-fragment reference energies
by +12.736456 and the summed ghost-fragment energies by +11.291338 kcal/mol.
Consequently the conditional no-CP/CP differences are -13.149553/-11.704435.
The dominant short-distance anomaly is in the subtracted reference terms.

The independent-monomer model retains a repulsive short-distance rise for
this example. This does not certify its quantitative accuracy or all twelve
model curves. The variable conditional reference is also a substantive basis
response diagnostic, rather than a small numerical roundoff effect.

## What the current CP plots mean

Standard CP compares own and ghost spaces with the selected orbital functions
held fixed; see the [DIRAC counterpoise tutorial](https://www.diracprogram.org/doc/release-22/tutorials/dft/bsse.html).
Case 15's CP curve is the conditional frozen-dimer-basis quantity
`E_AB - E_A(ghost; c_AB) - E_B(ghost; c_AB)`. It characterizes borrowing within
that selected basis. It is not identical to adding its BSSE value to
`model_interaction_kcal_mol`.

A model interaction plus a conditional-CP correction would, algebraically,
retain the reference term above: `conditional_CP + delta_ref`. That is a
separate explicitly defined composite estimate and has not been adopted or
validated as a standard CP-corrected model potential here. It must not be
introduced merely to change the apparent curve shape.

Likewise, equal fixed/adaptive interaction values at scale 1.00 were an
acceptance condition only for the existing conditional diagnostic. With
policy-specific independently defined monomer limits, identical dimer bases
alone do not require identical binding-energy zeros.

## Independent numerical check

PySCF 2.9.0 independently reproduced 15 native energies: dimer, both own and
both ghost fragments for adaptive/fixed scale 0.80 and the shared scale-1.00
basis. The same exported Gaussian definitions, real nuclear charges, spherical
AOs and common dimer DF metric were used. Maximum converged-energy difference
was **1.5916e-11 Eh**; every reproduced solution passed real internal stability.
All MINAO starts converged. Three exploratory core-only starts did not converge
and were retained as rejected starts, never used as energies. The check does
not prove global HF uniqueness. No finite differences or complete AO ERI were
used. PySCF is an independent comparison only, not a production fallback.

See `independent-check.json`, `hbr-adaptive-minus-fixed.csv`,
`reference-shifts-all-systems.csv`, `reference-shift-summary.csv` and the
two-panel `reference-decomposition.pdf` for reproducible evidence. The largest
conditional reference variations also occur for HF–NH3 and HCl–NH3, so this
interpretation issue extends beyond the supplied HBr–NH3 example.
