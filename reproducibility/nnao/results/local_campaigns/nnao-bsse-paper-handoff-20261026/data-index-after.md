# NNAO computational examples: current data index

Data-only handoffs for the manuscript/figure thread. Source records, units,
numerical status and limitations are documented in each dated package.

| Scope | Current package | Relationship to older data |
|---|---|---|
| BSSE / frozen-basis counterpoise, focused Case 14 | [2026-10-26 BSSE supplement](2026-10-26/cases/14_bsse_counterpoise/README.md) · [figure gallery](2026-10-26/cases/14_bsse_counterpoise/index.html) | Reorganizes unchanged Case 07/08 data; adds full-22/common-12 S22 summaries and direct BSSE–distance plots |
| Ethane C-C rigid stretch, Case 05 | [2026-10-22 corrected Case 05](2026-10-22/cases/05_ethane_cc_stretch/README.md) | Supersedes 2026-10-21 Case 05: corrected 56-point MACE/free curves, seven conventional bases, radial functions and 112 PySCF checks |
| Other computational examples, Cases 01–04 and 06–13 | [2026-10-21 archive](2026-10-21/README.md) | Unchanged |
| Frozen update-3351 inference model | `2026-10-21/models/update-3351/` | Shared by both handoffs; not retrained or duplicated |

For the corrected ethane case, use [504 curve values](2026-10-22/cases/05_ethane_cc_stretch/artifacts/nnao-ethane-dissociation-benchmark-20261022/curves.csv),
[56 geometries and final-basis mapping](2026-10-22/cases/05_ethane_cc_stretch/structure-index.json),
and the [flat PySCF comparison](2026-10-22/cases/05_ethane_cc_stretch/tables/pyscf-comparison.csv).
The previous five-point gap and large hcore-branch jump are superseded.

The historical `2026-10-21/` archive and its checksums remain unchanged.
`output/ethane_cc_stretch/` still uses the older data until the figure thread
explicitly regenerates it. This handoff does not edit the manuscript, figures
in `output/`, or training state.
