# NNAO computational examples: current data index

Data-only handoffs for the manuscript/figure thread. Source records, units,
numerical status and limitations are documented in each dated package.

| Scope | Current package | Relationship to older data |
|---|---|---|
| Ethylene total-density planes, Case 17 | [2026-10-27 six-basis comparison](2026-10-27/cases/17_ethylene_density_planes/README.md) | xy molecular and xz perpendicular planes; 12 maps; update 3351 MACE/free + four conventional bases; matched DF-RHF and shared color scale |
| Ethane C0 s/p/d radial response and all contraction coefficients, Case 16 | [2026-10-27 analysis](2026-10-27/cases/16_ethane_spd_radial_response/README.md) | 25 torsion angles + corrected 56 stretches; update 3351 MACE vs free contraction; three channel figures; no new SCF |
| Dimer CP × fixed/adaptive MACE output, Case 15 | [New controls](2026-10-26/cases/15_dimer_cp_output_controls/README.md) · [two-group figures](2026-10-26/cases/15_dimer_cp_output_controls/figures/index.html) | Corrected constant isolated-monomer references; model + frozen-parameter CP estimate; conditional-v1 exports archived |
| 500-structure training pool | [Dedicated training dataset](../training/nnao500-20261021/README.md) | Exact family500/update-3351 input; 500 XYZ files, combined XYZ, original JSON, coordinate tables and lineage |
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
