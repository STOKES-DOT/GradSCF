# Sixty-system SCF fixture catalog

This directory contains **inputs, not calculated results**. The primary catalog
has60 distinct physical systems, with12 assigned to each of STO-3G,6-31G*,
def2-SVP,cc-pVDZ and def2-TZVP. This is60 primary system/basis cases, **not**
60 systems multiplied by five bases. It spans5--150 atoms, counting hydrogens.
No SCF calculation, integral evaluation or molecular optimization was executed
while preparing these fixtures.

Canonical executable inputs and geometries are versioned. Raw source responses
under `sources/` are an ignored download cache; `prepare_catalog.py` can recreate
them. Their URLs and content hashes remain recorded in the catalog/manifest.

## Files and sources

- `systems.json`: executable input data: `id`, `atom` (Angstrom), `basis`,
  `charge`, `spin=N_alpha-N_beta`, formula/counts, and source provenance.
- `catalog_summary.csv`: one row per primary case for inspection.
- `catalog_seed.json` and `prepare_catalog.py`: the reproducible source selection
  and standard-library/curl preparation procedure. Existing responses are cached.
- `sources/` and `geometries/`:49 PubChem identity/3D records, six RCSB CCD
  identity/ideal-coordinate records, and five explicitly constructed ideal
  cation-water inputs. The original144-atom RAP coordinates are
  additionally retained as the parent of the150-atom constructed system.
- `fixture_manifest.json`: counts, element coverage, selected cross-basis controls
  and hashes. Each system also hashes its identity response and geometry.
- `sources/unavailable/`: identity lookups for ten rejected PubChem candidates,
  with the actual HTTP404 3D endpoint outcome. Their missing coordinates were
  not replaced by2D drawings or guessed geometries.
- `sources/replaced/`, `geometries/replaced/` and
  `sources/coordination_replacements.json`: the five redundant organic inputs
  replaced before benchmark timing to add the missing reactive early-period
  elements, preserving the12-per-basis allocation.

PubChem names were resolved with its primary PUG-REST identity service, then
the returned CID was used to request a3D SDF. Formula, formal charge, explicit
hydrogens, atom count and electron/spin parity were checked. RCSB metadata were
retrieved from `data.rcsb.org/rest/v1/core/chemcomp/<id>` and ideal coordinates
from `files.rcsb.org/ligands/download/<id>_ideal.sdf`; the formula and charge
were independently checked against the SDF. These are computed/ideal structures,
not experimental coordinates or newly optimized quantum-chemical geometries.

The six CCD records are ATP, NAD, FAD, BO3 (boric acid), BOR (the specified
arginine boronic acid) and RAP (rapamycin). The catalog uses each CCD record's
**explicit protonation/formal charge**, including net-neutral ATP/NAD/FAD;
it does not infer physiological aqueous charge states from familiar names.

## Ideal cation-water controls

These five explicitly declared **unoptimized computational inputs** add Li,
Be, Na, Mg and Al. They are not experimental coordination structures, optimized
aqueous ions, or conversions of disconnected PubChem salt records.

| Replaced primary input | New input | Basis | Charge/spin | Atoms/electrons |
|---|---|---|---|---|
|decane|Li+(H2O)2|STO-3G|+1/0|7/22|
|ethylbenzene|Be2+(H2O)2|6-31G*|+2/0|7/22|
|anisole|Na+(H2O)2|def2-SVP|+1/0|7/30|
|acetaldehyde|Mg2+(H2O)4|cc-pVDZ|+2/0|13/50|
|dimethyl ether|Al3+(H2O)4|def2-TZVP|+3/0|13/50|

`geometry_construction.py` reuses the same archived water as the150-atom
benchmark. A proper rigid rotation aligns each water H--O--H bisector away
from the cation. The metal is at the origin. Two oxygens use opposite axial
directions; four oxygens use normalized tetrahedral directions. Every M--O
distance is fixed to the **chosen benchmark input2.4 Angstrom**, without any
claim that this is an experimental or optimized bond length.

The XYZ coordinates retain12 decimal places. After serialization, internal
water pair distances agree with the archived geometry to8.1e-13 Angstrom,
M--O distances agree to1.8e-13 Angstrom, outward bisector cosines are at least
0.9999999999999999, and the minimum interatomic distance is0.9579 Angstrom.
The source descriptors contain these checks, construction-code/water hashes,
and the assigned bundled basis snapshot/hash. Source inspection verified metal
shells for each assignment; runtime integral/basis evaluation remains pending.

## The150-atom endpoint

`rapamycin-two-water-capacity` consists of the verified144-atom RCSB RAP ideal
structure plus two archived neutral water molecules. This is an **unoptimized
rigid capacity benchmark**, not an experimental hydrate or a claimed bound
complex. Its formula is C51H83NO15; charge0, spin0.

The archived water input is
`reports/tddft_perf/20261027-33a7cab/cases/water-pbe-tda.json`, with the same
Angstrom/neutral/singlet convention as that probe. Both waters retain their
internal geometry. Their oxygen positions are rigidly translated to
`(x_max(RAP)+8,0,0)` and `(x_max(RAP)+16,0,0)` Angstrom. All cross-fragment
atom distances are checked to exceed5 Angstrom. Parent and derived geometry
hashes, water-input hash/settings, translations and measured minimum separation
are stored in the system's `source` object. No coordinate optimization occurs.

The algebraic STO-3G size estimate is418 spatial AOs. Its s4 ERI array alone
would occupy about61.35GB in float64, before grids/workspace. This is a
preflight estimate, not a measured allocation: the runner must determine actual
AO dimensions and obey its resource/memory gate, especially for GHF/GKS.

## Coverage and comparison rules

The covered elements are H,Li,Be,B,C,N,O,F,Na,Mg,Al,Si,P,S,Cl: all15 reactive
elements through Ar. Only the noble gases He,Ne,Ar are unrepresented. Their
absence is a geometry-coverage boundary, not a basis-set or integral limitation.

Five explicit doublet cases are vertical radical ions: phenanthrene cation,
pyridine cation, nitrobenzene anion, styrene cation and p-benzoquinone anion.
They use the documented neutral source geometry at fixed nuclei, with changed
charge and spin; no ionic-geometry optimization or adiabatic comparison is
claimed. Neutral phenanthrene and its cation are distinct charge-state systems,
not two conformers counted as different species. Spin specifies the collinear
seed/occupation target; it does not certify an S-squared eigenstate or stability.

DFT settings are float64, Cartesian AO conventions and grid level3. The matched
functional set is SVWN (`lda,vwn` in PySCF), PBE, PBE0 and CAM-B3LYP. Original
wB97X-D remains in the existing separate regressions; the main matched set
avoids adding a dispersion-definition mismatch. These settings are recorded
in `systems.json` but are not results or evidence of runtime support.

RHF/RKS are not applicable to the declared doublets. Current GKS RSH and
GHF/GKS auxiliary-DF/direct paths are unsupported and must be recorded as such;
UKS must never be substituted for a requested spinor calculation. Supported
paths can still be resource-limited or nonconverged, which are separate outcomes.

The primary assignments deliberately put larger systems in the smaller basis
and reserve def2-TZVP for manageable molecules, including phenanthrene. Do not
fit an asymptotic scaling exponent across this heterogeneous table. Suggested
independent cross-basis controls are methane in all five bases, benzene in
6-31G*/def2-SVP/def2-TZVP, and phenanthrene in def2-SVP/def2-TZVP. These add at
most seven cases beyond their primary assignments and are reported separately,
not expanded into the whole60-by-five product.

## Data-only reproduction

From a source checkout, with network access only for missing public files:

```bash
python benchmarks/probes/scf_solver/prepare_catalog.py
python benchmarks/probes/scf_solver/prepare_catalog.py --validate-only
```

Validation checks60 IDs, the12-per-basis allocation, source hashes, formulas,
electron/spin parity, actual SDF/XYZ/argument coordinate equality, rigid water
invariants and the5/150
endpoints. It does not import JAX, PySCF or GradSCF. All quantum calculations,
solver comparisons, numerical derivatives and performance measurements belong
to the root-controlled remote Slurm runner and are still pending for this catalog.
