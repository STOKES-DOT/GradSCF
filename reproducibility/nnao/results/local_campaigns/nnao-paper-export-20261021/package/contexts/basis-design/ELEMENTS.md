# NNAO elements and single-zeta fixed cores

## Final core rule

The fixed core has **one contracted radial function per occupied atomic radial
orbital**. Its parameters are selected from original BSE 6-31G version-1 records:

- Inner s/p: **6 primitives → 1 radial contraction**.
- Ga-Kr inner 3d: **3 primitives → 1 radial contraction**.
- The separate one-primitive d block in the parent 6-31G data is **not** added to
  the fixed core. No coefficient combination, fitting, padding or ECP is used.
- The model's existing independent d polarization channel is unchanged. All m
  components of a shell share its radial coefficients.

This is the chosen compact NNAO core derived from 6-31G records, not a claim that
we retain the full parent 6-31G contraction pattern. The original BSE file is stored
verbatim in `data/bse_631g_v1.json`; `data/bse_sources.json` records its URL, version,
hash, and the explicit `bse-6-31g-sz-core-sp6-d3-v1` selection policy.

For Ga-Kr the fixed core is `3s2p1d` (14 spherical AOs):

| Model | Fixed core | Valence | Polarization | Total spherical AOs |
| --- | --- | --- | --- | ---: |
| SZP | 3s2p1d | 1s1p | 1d | 23 |
| DZP | 3s2p1d | 2s2p | 1d | 27 |
| TZP | 3s2p1d | 3s3p | 1d | 31 |

The same core is used for every S/M/L primitive tier and zeta level. All electrons
participate in SCF; fixed basis parameters do not freeze electrons.

## Scope and API

Grid families support these 26 first-four-period main-group elements:

`H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ga Ge As Se Br Kr`

The current 1000-structure dataset contains 25 of them; Ga has no qualified sample.
No transition metals or elements beyond Kr are added to this experiment.

Use `core_primitives='6-31g'`, the default for grid families. `'reference'` is an
alias for this single-zeta rule. Numeric `3` and `6` retain explicit legacy snapshot
meanings; they are not the setting for this new experiment. The legacy
`supported_elements()` default describes `szp3`; request a grid family for its scope.

```python
from gradscf.model.nnao import (
    prepare_direct_basis, model_config_for_elements, exponent_keys_for_elements,
)
from gradscf.model.nnao.config import EXPANDED_ELEMENTS

config = model_config_for_elements(EXPANDED_ELEMENTS, basis_family='nnao_szp_l')
layout = prepare_direct_basis('H 0 0 0; Br 0 0 1.414',
    basis_family=config['basis_family'], core_primitives='6-31g')
keys = exponent_keys_for_elements(EXPANDED_ELEMENTS, basis_family=config['basis_family'])
```

Element order is part of model/checkpoint identity. Stage-C reference JSON must
declare the vocabulary explicitly; omitting it retains the original CHNOF model.
`basis_grid run --pilot-references ...` requires references covering every training
element. Historical CHNOF joint-training CLIs keep their fixed configuration.

## Candidate and cache invalidation

Candidates made under 3-21G cores or the abandoned two-d-core interpretation must
be recalibrated. Candidate binding checks both the core selection policy and BSE
asset hash, in addition to primitive order and absolute active exponents. Cache
identity includes the selected core policy and its actual matrices. Old caches
cannot silently be read under this interpretation, even for H-only examples.

S/M/L active-pool budgets and polarization initialization retain the existing
design. These are starting pools; A/B calibration and free-contraction assessment
still precede a formal fixed-exponent MACE run.

## Verification boundary

Tests check original coefficients, one fixed core d radial function, all nine
layouts, positive overlap spectra, native versus projected S/h/DF factors and
coefficient VJPs, stale candidate/cache rejection, real MACE and implicit-SCF
training/restoration. Data availability is not a complete SCF or DF-accuracy
benchmark for every new molecule.

No scalar-relativistic Hamiltonian is introduced. The native integral library is
unchanged, and production code does not call PySCF. Running original job 96510
keeps its frozen source; new training remains paused while this design is verified.

Source: [BSE 6-31G v1](https://www.basissetexchange.org/api/basis/6-31g/format/json/?version=1).
The [BSE Pople notes](https://www.basissetexchange.org/family_notes/pople/) document
Ga-Kr version differences from Gaussian; the exact pinned asset controls this
implementation, rather than the basis name alone.
