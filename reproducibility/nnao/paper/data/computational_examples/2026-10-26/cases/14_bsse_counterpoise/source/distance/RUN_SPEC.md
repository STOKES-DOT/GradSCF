# NNAO distance curves and reaction barriers

## Fixed protocol

Frozen MACE update 3351, checkpoint SHA256
`a7144d211a7b3d9fc4cb6899bf930072992e073bbc79d8cae282611682bc7c1f`.
Native all-electron DF-HF/def2-universal-jkfit, spherical AOs, float64;
RHF for spin zero and UHF for explicitly specified nonzero spin. Source
charge/multiplicity is mandatory. No ECP, geometry optimization, gradient
test, ZPE, thermal correction, or entropy term is introduced.

Use existing native SCF adapters and convergence thresholds from
`tests/comparisons/nnao_core/native.py`: energy 1e-12 Eh, density 1e-10,
gradient 1e-9, max 200 cycles, AO commutator <=1e-7, overlap eigenvalues >1e-10.
Bounded shifted rescue must finish with a converged zero-shift calculation.
Record UHF spin contamination. This validates stationary HF branches, not
global orbital minima. Charged and radical MACE inputs extrapolate beyond
the neutral closed-shell training domain.

## Source and representative selection

Author-maintained data: https://github.com/aoterodelaroza/refdata,
commit `2b09de29c7e0c23e621b2de988bc41c8090d4481`.
Every selected XYZ is checked against its Git blob SHA1; source paths and
SHA256 values are retained. The source archive license is preserved in raw/.
Dataset definition is `dataset.json`, with all excluded cases recorded.

- **S66x8**: six complete eight-point curves (48 points): Water-Water,
  AcOH-AcOH, Benzene-Benzene_pi-pi, Cyclopentane-Cyclopentane,
  Benzene-Benzene_TS and Benzene-Water_OH-pi. Two H-bond, two dispersion
  and two mixed/orientation examples; not the full 528-point benchmark.
- **HB300SPXx10**: six complete ten-point curves (60 points): source groups
  1.01, 1.09, 1.15, 1.22, 1.28 and 2.10. Cover P, S, F, Cl and Br.
  52 of the full 300 equilibrium complexes contain unsupported model elements
  (I); do not replace these elements or extrapolate an output head.
- **BH9**: one smallest supported reaction per original numeric class,
  chosen by maximum species atom count, total nuclear charge, then source
  reaction number. Selected: 01_1, 02_134, 03_1, 04_1, 05_1, 06_1,
  07_1, 08_1 and 09_1 (34 species, maximum 17 atoms). Among all 449
  source reactions, 184 meet the initial panel's supported-element and
  maximum-30-atoms-per-species requirements. The others are listed with reasons.

All 449 BH9 reaction definitions were checked for element and charge
conservation. Forward and reverse reference barriers close with the reference
reaction energy within 0.01 kcal/mol, reflecting independent published rounding.
No claim of training-set or chemical-family independence is made.

## Numerical comparisons

Six bases: STO-6G, 6-31G(d,p), def2-SVP, cc-pVDZ, cc-pVTZ and NNAO/MACE.
The bundled snapshot lacks Br/6-31G(d,p); that single definition is supplied
explicitly by Basis Set Exchange version 1 (revision 2018-06-19), with its
original general/SP contractions, no ECP and no substituted exponents.
The native HBr/6-31G(d,p) energy agrees with independent same-shell PySCF
DF-HF to 4.55e-12 Eh on local CPU. All 50 element/basis combinations were
checked before the representative submission.
The large QZ control that exhausted GPU memory in the preceding suite is
not required for this representative panel. HF/cc-pVTZ is a finite-basis
same-method anchor, not a correlated/CBS limit.

For NCI, report separately:

1. Actual model interaction: E_AB(c_AB)-E_A(c_A)-E_B(c_B), with independent
   MACE predictions for dimer and isolated monomers. Rigid monomer geometries
   are checked against the corresponding upstream dimer atom blocks.
2. Frozen-dimer-basis CP diagnostic: all five energies use c_AB and the
   common dimer auxiliary metric, with ghosts excluded from neural inputs.
   This is not a CP correction to the independently predicted model energy.

Primitive/DF numerical factors remain three-index or packed; no four-index
ERI is stored. Isolated monomer energies are cached at their exact source
geometry. Failed cases remain explicit. Resume requires identical inputs
and benchmark source hashes.

For BH9, sum every reactant/product fragment with its signed stoichiometry:
forward = E_TS-E_R, reverse = E_TS-E_P, reaction = E_P-E_R.
Report MAEs against both same-method HF/cc-pVTZ and the published
DLPNO-CCSD(T)/CBS reference; NCI correlated references include physics missing
from HF, particularly dispersion, so their errors are not pure basis errors.

## Figures

- NCI interaction energy vs original distance scaling, all original points,
  with correlated reference and traditional basis controls. Raw model and
  fixed-dimer CP results use separate figures. Missing points remain gaps.
- BH9 stationary-state energy diagrams: horizontal R/TS/P levels, each
  method's reactants at zero, schematic connectors, and arrows for forward
  and reverse barriers and reaction energy. Include per-reaction numeric
  tables and a nine-panel overview. These are not IRC trajectories or
  activation free energies. Preserve negative/submerged barriers.
- CSV files retain all six basis results; the main figures emphasize
  6-31G(d,p), cc-pVDZ, cc-pVTZ, NNAO and the correlated reference.

## Work and resources

Local gates: eleven new protocol tests plus sixteen reused native/core checks;
an additional independent numerical check covers the BSE Br asset.
The short profile covers three NCI geometries (water, P and Br) and three
BH9 categories (radical, closed-shell and anionic); 39 cases across STO-6G,
cc-pVTZ and MACE, about 93 SCF solves. Request one V100, 32 CPU, 128 GiB,
maximum ten minutes.

The representative panel has **852 cases** (648 dimer/basis cases and
204 reaction-species/basis cases), plus 144 cached isolated-monomer solves;
approximately **3588 primal SCF solves**, excluding bounded rescues.
Prepared request: one A100 80G, 32 CPU, 512 GiB, maximum three hours.
The user approved this representative request on 2026-10-20. It depends on
short job 102209 and verifies its numerical report is complete, 39 passed,
zero failed; Slurm exit status alone does not satisfy the gate.
No full-dataset or multi-GPU campaign is implied by this first panel.

## References consulted

- S66 original: https://doi.org/10.1021/ct2002946.
- Revised S66x8 references used by this source:
  https://doi.org/10.1039/C6CP00688D.
- HB300SPX and its x10 extension:
  https://doi.org/10.1021/acs.jctc.0c00715.
- BH9 and correction (source DIN includes the corrected definitions):
  https://doi.org/10.1021/acs.jctc.1c00694;
  author postprint: https://digibuo.uniovi.es/dspace/handle/10651/62983.
- Reaction-progress energy diagram convention:
  *Visualizing Reaction Progress and the Geometry and Instability of the
  Transition State*, https://doi.org/10.1021/ed100977e.
  Plateau/arrow layout here is our implementation of the R/TS/P convention,
  not a copied figure or an assertion that an IRC has been calculated.

## Commands

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q \
  tests/nnao/test_reactivity_protocols.py \
  tests/nnao/test_core_benchmark_protocols.py tests/nnao/test_core_benchmark_native.py
python tests/comparisons/nnao_reactivity/prepare.py \
  --directory artifacts/nnao-curves-barriers-20261020
python tests/comparisons/nnao_reactivity/plot.py \
  artifacts/nnao-curves-barriers-20261020/representative \
  --dataset artifacts/nnao-curves-barriers-20261020/dataset.json
```
