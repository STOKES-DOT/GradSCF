# Data sources and attribution

## Training structures

- **GMTKN55**: selected structures from the Grimme-group database. The source
  metadata pins v2 to commit `ccabc16c8c4677fc60c19870ae96ac25405844a2`; earlier
  v1 imports retain their own identifiers. The upstream README identifies the
  dataset as CC BY 4.0 and the evaluation code as MIT:
  [pinned upstream README](https://github.com/grimme-lab/GMTKN55/blob/ccabc16c8c4677fc60c19870ae96ac25405844a2/README.md).
  Cite Goerigk et al., *Phys. Chem. Chem. Phys.* **19**, 32184–32215 (2017),
  [doi:10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G).
- **QH9Stable**: selected stable geometries, with original database row IDs and
  row hashes retained. The full Hamiltonian database is not mirrored here.
  Consult the [official QH9 source and paper](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9)
  for its original dataset citation and terms. The selected 500-system pool is
  not the original QH9 benchmark split.
- Exact dataset-source URLs, local original paths, selection rules, atom order,
  charge/spin and input checksums are preserved in each dataset's manifest and
  reference JSON. No geometry or state is approximated during collection.

## Tests and reference values

The copied case/protocol READMEs retain the original references for S22/S22B,
S66×8, HB300SPX, BH9, ISO34 and the other test subsets. The main basis-accuracy
comparison uses ISO34/W1-F12 and S22B/estimated CCSD(T)/CBS relative-energy
references. These are distinct from same-method large-basis absolute-energy
proxies. Refer to the saved reaction definitions and `REFERENCE_LEVELS.md`
files rather than inferring the reference method from the computed HF/DFT/MP2
method name.

## NNAO outputs and model provenance

Newly computed NNAO energies, densities, radial functions, coefficients,
training records and parameter arrays retain the producing run's source,
configuration and numeric identities. Third-party code/data notices remain
with their source files. The model-local MACE-JAX and native integral vendor
licenses are preserved in the source tree.

This directory is a prepared research data archive, not a separately minted
Zenodo/Figshare release. Cite the repository branch/commit plus the original
datasets when using it; add the paper/data DOI once one has been assigned.
