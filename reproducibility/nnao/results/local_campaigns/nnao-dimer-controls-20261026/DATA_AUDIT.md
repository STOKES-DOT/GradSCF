# Dimer CP × MACE-output controls: data audit

Status: the meaning of **fixed MACE output** requires clarification before
new numerical calculations. No calculation or remote job has been started.

Proposed interpretation: freeze each dimer's contraction coefficients at
distance scale 1.00 for the entire scan, versus recompute them at each scale.
Network weights and primitive exponents remain fixed in both cases. This
interpretation is not yet confirmed by the user.

| Figure group | Frozen at scale 1.00 | Geometry-adaptive MACE output |
|---|---|---|
| No CP | New fixed-output own-fragment interaction | Existing frozen-at-current-geometry own-fragment interaction |
| CP | New fixed-output ghost-fragment interaction | Existing current-geometry frozen-basis CP diagnostic |

For a valid CP comparison, the coefficients selected by either policy must be
kept unchanged among its five calculations at that geometry. Each uses one
common dimer auxiliary metric. Re-predicting fragment or ghost coefficients
inside the five-energy subtraction changes the basis spaces and requires a
different explicitly defined protocol.

Existing `model_interaction_kcal_mol` uses independently predicted isolated
monomers and is different from `uncorrected_interaction_kcal_mol`. These must
not be interchanged when comparing CP on/off. Molecular geometries, energy
zeros, auxiliary metric, HF restrictions and checkpoint must be matched.

Source: repaired complete update-3351 benchmark, 12 representative dimers,
108 geometry points. Adaptive MACE raw-frozen/CP and STO-6G raw/CP are already
available. At scale 1.00 the proposed fixed/adaptive policies coincide, leaving
96 new fixed-output five-energy cases plus 108 STO-6G+P cases (at most 1020
new SCF evaluations before any exact reuse). Other traditional bases remain
available from the same report. Full-run resources have not been requested.

STO-6G+P follows the existing CHNO control: original STO-6G plus a def2-SVP
p shell on H or d shell on heavy atoms. The exported candidate assets extend
this to all nine elements in the dimer dataset. For Br the original def2-SVP
d list contains occupied/core radial functions as well as the outer
uncontracted d function; the proposed control adds only the most diffuse
uncontracted d shell (exponent 0.389), preserving STO-6G's occupied shells.
No exponent is fitted or approximated. Assets are explicit in
`sto6g-p-reference-assets.json`; they have not been used for new calculations.
