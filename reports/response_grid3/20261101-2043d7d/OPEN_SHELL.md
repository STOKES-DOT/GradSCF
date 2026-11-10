# Fixed-geometry neutral/open-shell pair (2026-11-01)

Use **phenanthrene radical cation, C14H10+, charge +1, collinear seed spin 1**,
on the identical archived neutral PubChem CID 995 coordinates. Its 93 electrons
correspond to Nalpha=47, Nbeta=46 for the chosen doublet seed; the ideal pure-doublet
<S²> is 0.75. The neutral control has 94 electrons, charge 0, spin 0. Both have
14 carbon and 10 hydrogen atoms and exactly the same nuclear/AO/grid inputs.
This is a controlled fixed-nuclei vertical response and capacity benchmark.
It is not an optimized cation, an adiabatic excitation calculation, or a prediction
of experimental vibronic band origins. A requested seed spin does not prove the
converged determinant is the lowest stable state or an S² eigenstate.

The two independent chains are `DFT -> GW -> BSE` and `DFT -> TDDFT`.
Four XC choices, the GW variant, explicit RKS/UKS/GKS representation, and solver
choices belong to the runtime invocation rather than duplicated case files.

## Why this molecule

Bréchignac, Pino, and Boudin's primary gas-phase spectroscopy study explicitly
includes the phenanthrene cation, using photoionized aromatic/rare-gas complexes
and delayed photodissociation:
[2001 paper, DOI 10.1016/S1386-1425(00)00441-8](https://pubmed.ncbi.nlm.nih.gov/11345251/).
Halasinski, Salama, and Allamandola report neon-matrix electronic spectra and
identify phenanthrene-cation doublet transitions in section 4.2 and Table 2:
[ApJ 628, 555-566 (2005), author-hosted primary paper](https://www.astrochem.org/docs/Halasinskietal2005.pdf).
These references justify the chemical choice and electronic-state context; none
of their measured spectral positions is used as a target for this geometry.

Phenalenyl C13H9 is a credible similarly sized neutral doublet alternative.
However, Levey et al.'s primary experimental/computational work documents strong
Jahn-Teller and pseudo-Jahn-Teller coupling across multiple excited surfaces:
[ACS Phys. Chem. Au 5, 560-568 (2025), DOI 10.1021/acsphyschemau.5c00052](https://pmc.ncbi.nlm.nih.gov/articles/PMC12464889/).
Choosing phenanthrene+ here is an experimental-design decision: it preserves the
existing atom list, geometry, nuclear charge, AO counts, and quadrature inputs
against the neutral control. It is not a claim that phenanthrene+ is intrinsically
more accurate for TDDFT or BSE or free of vibronic/near-degeneracy effects.

## Geometry and basis provenance

`cases/phenanthrene-pubchem-995.sdf` is a byte-preserving copy from
`reports/hybrid_gw_geometry/20261031-f0c9b2b/cases/` at source commit
`2043d7d3eada53e2446dbb818a85f52aa52487b3`. Its SHA-256 is
`039ac4144c9910e1fe5a4900fb5a8aa6268c2c906fb1f5f64e5955936938081b`.
The archived origin is
[PubChem CID 995 computed 3D conformer](https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/995/SDF?record_type=3d).
Live PubChem retrieval was unavailable during this preparation; the existing
coordinates and hash were verified locally and have not been refreshed or
optimized. The SDF remains the neutral geometry source; charge/spin are explicitly
overridden in each JSON input. Coordinates are in Angstrom.

Cartesian shell counts in the versioned basis files give carbon/H AO counts
15/5 for def2-SVP and 36/6 for def2-TZVP. Thus the phenanthrene pair has 260/564
spatial AOs, respectively. Each JSON records source-file and exact atom-string
SHA-256 hashes. `weigend` remains the explicit RI auxiliary basis for continuity;
this comparison does not establish basis-set or RI-basis convergence.

## Full-space semantics and expected dimensions

All core and valence electrons, all occupied/virtual orbitals, all G/W orbitals,
and all GW quasiparticle targets are retained. The previous six-frontier-orbital
`orbs` list is deliberately absent. The number of returned low roots may be
limited by a solver request while its orbital/transition space stays complete.
Counts below are algebraic metadata, assuming no overlap-rank loss; assert against
actual runtime occupation and coefficient arrays before any large calculation.

| System | AO | RKS occupied/virtual | UKS occupied alpha/beta | UKS same-spin transitions | Whole GKS occupied/virtual | Whole GKS transitions |
|---|---:|---|---|---:|---|---:|
| Phenanthrene / SVP | 260 | 47 / 213 | 47 / 47 | 20,022 | 94 / 426 | 40,044 |
| Phenanthrene+ / SVP | 260 | not applicable | 47 / 46 | 19,855 | 93 / 427 | 39,711 |
| Phenanthrene / TZVP | 564 | 47 / 517 | 47 / 47 | 48,598 | 94 / 1,034 | 97,196 |
| Phenanthrene+ / TZVP | 564 | not applicable | 47 / 46 | 48,127 | 93 / 1,035 | 96,255 |
| H2O / SVP | 25 | 5 / 20 | 5 / 5 | 200 | 10 / 40 | 400 |
| OH / SVP | 20 | not applicable | 5 / 4 | 139 | 9 / 31 | 279 |

RKS transition dimensions are 10,011 (SVP phenanthrene), 24,299 (TZVP
phenanthrene), and 100 (SVP H2O). A whole GKS MO coefficient matrix has shape
`(2*nao, 2*nao)`, 0/1 occupations and `nelectron` occupied spinors. Its complete
occupied-to-virtual space includes the spin-flip sectors in a collinear limit.
`SpinBSESpace` in `src/gradscf/bse/space.py` currently describes two **spin-conserving
UKS blocks**; its presence is not evidence of whole-spinor GKS support.
The full X/Y response operator has twice the transition dimension; TDA is a
separate approximation and must not silently replace either requested full chain.

Every stage must use grid level 3. Record actual point count, coordinates, weights,
ordering, pruning, and source hash: a level label alone does not prove equal
quadrature. No point count is guessed in the fixtures.

## Required result qualifications

- Report SCF convergence and stationarity, actual electron count/occupations,
  <S²> (target 0.75 for the doublet), spin expectation/magnetization for GKS,
  and the initial density/seed. UKS/GKS can converge to different broken-symmetry
  solutions; convergence alone does not establish orbital stability or absence
  of spin contamination. If stability is not tested, explicitly record that.
- Report response residuals and all zero, negative, or complex roots encountered.
  A small/unstable mode needs classification; never conceal it by switching to
  TDA, discarding a channel, or shrinking orbital windows. Whole-spinor response
  can include spin-rotation modes, so an unconditional positive-root gate is
  insufficient to distinguish symmetry modes from numerical failure.
- Check every requested GW QP convergence flag and the finite energy arrays.
  Full-QP completion is separate from full G/W summations. Do not fill missing
  QPs with uncorrected KS values and call it a completed all-space GW->BSE run.
- Def2-SVP/TZVP and fixed weigend have no diffuse augmentation. High virtual or
  Rydberg/continuum-like roots and spectral accuracy require separate convergence
  work. No optimization, SOC, vibronic corrections, solvent, or experimental
  equivalence is asserted by these metadata files.

H2O/OH geometries exactly reuse `tests/gw/test_gw_cd_ugw.py` and are promoted to
Cartesian def2-SVP/grid3 solely for small capability checks. OH's documented
STO-3G zero-root issue in `tests/test_streamed_scf_tddft.py` is a caution about
oracle selection, not a measured result for the new basis/grid. Keep H2O/OH
capability results separate from the phenanthrene capacity and physics results.

Fixture preparation performed only text/data/hash and electron/dimension
consistency checks. No quantum-chemical calculation or remote job was run.
