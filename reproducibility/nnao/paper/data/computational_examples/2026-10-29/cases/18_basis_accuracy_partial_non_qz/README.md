# Case 18: non-QZ basis accuracy against database references

Data-only handoff for NNAO_paper. No manuscript or existing manuscript figures are edited.

## Scope

Partial snapshot of the 27-basis campaign: **24 complete bases** (15 conventional and all nine NNAO), **30 physical structures**, **1440 energy records**, and **12 database reference energy differences**. The excluded bases are def2-QZVP, def2-QZVPP, cc-pVQZ, and the aug-cc-pVQZ finite reference.

Snapshot: 2026-10-29T17:07:44.104731 (cluster local time, UTC+08:00).

All 30 structures contribute: 12 structures form six ISO34 isomerizations; six S22 dimers plus their 12 independently evaluated fragments form six interactions. The database does not supply 30 absolute electronic-energy reference values here. Absolute computed energies are preserved without presenting an unsupported absolute-energy accuracy score.

## Two independent figures

- [DF-MP2 PNG](figures/accuracy-vs-ao-mp2.png) · [PDF](figures/accuracy-vs-ao-mp2.pdf) · [SVG](figures/accuracy-vs-ao-mp2.svg)
- [omegaB97M-V PNG](figures/accuracy-vs-ao-wb97m-v.png) · [PDF](figures/accuracy-vs-ao-wb97m-v.pdf) · [SVG](figures/accuracy-vs-ao-wb97m-v.svg)

Both figures have identical axis limits and the same bubble-size mapping. Every point represents one basis and one method; no structure, isomerization, or interaction is dropped within an included basis.

- x: arithmetic mean of contracted spherical AO counts over all 30 structures.
- y: MAE in kcal/mol against the frozen ISO34 and S22B database references, on a logarithmic axis.
- Bubble area: arithmetic mean number of unique spherical primitive GTO functions over the same 30 structures. Core functions and angular degeneracy are included; repeated contraction columns do not duplicate a primitive. Radial primitive counts are also retained in basis-costs.csv.
- NNAO labels omit the NNAO prefix in the plots; SZP-S means NNAO-SZP-S. Colors identify basis families.

The y value is 0.5 × (MAE of six isomerizations + MAE of six dimer interactions). Since each category contains six observations, this also equals the MAE of all 12 differences. Interactions are E(AB) − E(A) − E(B), without CP or BSSE correction.

## Reference calculation levels

| Database subset | Selected values | Reference level | Primary source |
|---|---:|---|---|
| ISO34 | 6 | **W1-F12 composite wavefunction method**, updated in GMTKN55 | [Goerigk et al., PCCP 19, 32184–32215 (2017), section 2.2.6](https://doi.org/10.1039/C7CP04913G) |
| S22B | 6 | **Estimated CCSD(T)/CBS**, using an MP2/CBS focal-point result plus a CCSD(T)-MP2 correction | [Marshall, Burns and Sherrill, JCP 135, 194102 (2011)](https://doi.org/10.1063/1.3659142) |

The ISO34 values replace the older back-corrected experimental references.
W1-F12 is a composite protocol, not one finite-basis CCSD(T) calculation.
The S22B label retains “estimated” rather than implying a direct infinite-basis
CCSD(T) calculation. Both reference levels are printed in each figure.
All 12 numerical values have been checked against the original .res files at
upstream commit ccabc16c8c4677fc60c19870ae96ac25405844a2.

See [reference table](tables/reference-energies.csv),
[reference-level details](REFERENCE_LEVELS.md), and [BibTeX](references/references.bib).
Original .res and .bib files are retained under references/. Every row in
relative-energies.csv also identifies its reference method and DOI.

## Nine NNAO results

| Basis | Mean AO | Mean primitive GTOs | DF-MP2 MAE (kcal/mol) | omegaB97M-V MAE (kcal/mol) |
|---|---:|---:|---:|---:|
| NNAO-SZP-S | 45.867 | 117.867 | 3.360106 | 2.843319 |
| NNAO-SZP-M | 45.867 | 133.333 | 2.499008 | 1.502590 |
| NNAO-SZP-L | 45.867 | 177.600 | 3.573521 | 1.642328 |
| NNAO-DZP-S | 61.333 | 117.867 | 2.159292 | 3.015409 |
| NNAO-DZP-M | 61.333 | 133.333 | 2.412669 | 1.568990 |
| NNAO-DZP-L | 61.333 | 177.600 | 1.152439 | 0.435237 |
| NNAO-TZP-S | 76.800 | 117.867 | 2.427427 | 3.101244 |
| NNAO-TZP-M | 76.800 | 133.333 | 1.698585 | 1.502205 |
| NNAO-TZP-L | 76.800 | 177.600 | 0.896771 | 0.460941 |

Within this selected set, NNAO-TZP-L has the lowest NNAO DF-MP2 MAE (0.896771 kcal/mol); NNAO-DZP-L has the lowest NNAO omegaB97M-V MAE (0.435237 kcal/mol). The latter is close to 6-311++G(2d,2p) (0.434802 kcal/mol), with mean AO counts 61.333 and 120.000, respectively. The primitive counts differ, so this is an AO-size comparison rather than a measured runtime claim.

## Numerical definition and provenance

- GPU4PySCF 1.7.0, PySCF 2.8.0, CuPy 13.4.1, cuTENSOR 2.2.0; effective CUDA 12.8 from NVHPC 25.3.
- GPU records were computed on Tesla V100-SXM2-32GB and NVIDIA A800. Per-record device, job ID and elapsed time are retained; no mixed-device timing ranking is made.
- Neutral, closed-shell H/C/N/O structures; spherical AOs and float64. SCF tolerances: 1e-10 Eh energy, 1e-7 orbital gradient; at most 200 cycles, with final unshifted convergence required.
- DF-MP2 uses RHF orbitals, freezes one 1s MO per real C/N/O, uses FP64, and does not save T2. J/K auxiliary basis: def2-universal-jkfit; correlation auxiliary basis: aug-cc-pVQZ-ri.
- omegaB97M-V includes its nonlocal correlation. Both main and NLC grid levels are 3.
- No CP/BSSE, ghost-atom, conditioned-monomer, CPU/GPU comparison, new SCF, or model training was performed to make this handoff. It only processes existing physical results.
- Published reference values and their source definitions/sign conversions are frozen in inputs/observables.json. S22 interaction values are the negatives of the source database dissociation energies.
- This is combined method-and-basis accuracy against database references, not a pure basis error relative to the same-method basis limit. QZ-dependent basis-limit and absolute-energy error panels are not part of this snapshot.
- Methane fragment sources S22/08/A and S22/08/B are related to training source GMTKN55-v2/S22/08a. This is recorded in inputs/selection-summary.json; no independent-holdout claim is made.
- NNAO checkpoint updates, parameter/source/checkpoint hashes and complete explicit atom-wise basis shells are retained in frozen/*.json. The checkpoint binaries are not duplicated; the exported shells fully specify the basis used for these energy records.

## Data and reproduction

- tables/summary.csv: 48 basis/method scores, including separate isomer and dimer MAEs.
- tables/relative-energies.csv: 576 predicted/reference energy differences and signed/absolute errors.
- tables/total-energies.csv: all 1440 absolute electronic-structure energies in Hartree, plus sizes/hardware/record links.
- tables/basis-costs.csv: the common 30-structure AO and primitive averages, and NNAO lineage.
- structures/: 30 XYZ files and all-30.xyz; coordinate units Angstrom.
- results/: original physical-result JSON files only. snapshot-manifest.json contains original file hashes.
- frozen/: all 24 explicit basis definitions. inputs/: original structures, reference definitions and settings.
- run_gpu_engine.py and protocols.py: unchanged numerical-source archives used to verify energy identity; build_partial.py does not import the GPU engine.
- verification.json records completeness. Figure layout coordinates are in figure-layout.json.

Reproduce tables and the two independent figures with Python, NumPy and Matplotlib:

```bash
python build_partial.py
```

The original inputs/settings.json is retained byte-for-byte for lineage. Its legacy CP diagnostic upper-bound field is inactive; the snapshot scope is authoritative and cp_bsse_included is false.
