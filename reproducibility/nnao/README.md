# NNAO paper data and reproducibility archive

This directory is the data release preparation area for the `gradscf_nnao`
branch. It contains the collected NNAO training structures, scientific test
outputs, inference parameters, figures, configurations, and provenance. No new
SCF calculation or model training is performed by the archival tools.

## Start here

| Content | Location |
|---|---|
| Exact ordered 500-system training input, 500 XYZ files and coordinate tables | [paper/data/training/nnao500-20261021/](paper/data/training/nnao500-20261021/README.md) |
| Original 1000-system candidate selections, element-scope revisions, and 500-system selection | [datasets/selection_archives/](datasets/selection_archives/) |
| All located 24/50/200/500-system definitions | [dataset-catalog.json](dataset-catalog.json) |
| Nine selected SZP/DZP/TZP × S/M/L trained models | [models/canonical-nine.json](models/canonical-nine.json) |
| Exact nine-model parameter hashes used by the basis-accuracy benchmark | [models/benchmark-nine.json](models/benchmark-nine.json) |
| Searchable model configurations, sources, parameter hashes and paths | [models/catalog.csv](models/catalog.csv) |
| Training logs, histories, diagnostics and calibration results from local workspaces | [results/local_campaigns/](results/local_campaigns/) |
| Final collected server-side training and test outputs | [results/remote_campaigns/](results/remote_campaigns/) |
| Curated paper cases and their corrected/historical relationships | [paper/data/computational_examples/README.md](paper/data/computational_examples/README.md) |
| Figure-generation scripts and saved scientific figures | [paper/scripts/](paper/scripts/), [paper/output/](paper/output/) |
| Full saved 27-basis MP2/ωB97M-V comparison, including QZ | [results/remote_campaigns/NNAO_basis_accuracy_20261029/analysis/](results/remote_campaigns/NNAO_basis_accuracy_20261029/analysis/) |
| Hidden L=1/L=2 matched 50-system experiment | [hidden_l50/results-20261031/](hidden_l50/results-20261031/RESULTS.md) |
| Fixed S/M/L primitive exponent pools | [pools/](pools/) |
| File identities and collection scope | [data-manifest.json](data-manifest.json), [archive-plan.json](archive-plan.json), [remote-plan.json](remote-plan.json) |
| Source version and verification | [source-manifest.json](source-manifest.json), [VERIFICATION.md](VERIFICATION.md), [DATA_VERIFICATION.json](DATA_VERIFICATION.json) |

The 500-system input is 400 GMTKN55-v2 plus 100 QH9Stable structures, with 25
observed elements, 1–30 atoms per structure, and 5,402 atoms in total. Its
untouched reference JSON has SHA256
`5e1b21c837f318da1bedc0b55f3f2bbfefc45218390928ce1b217db0607dc299`.
Older candidate selections with broader element vocabularies are retained as
history; they are not assertions of supported chemistry or completed training.
The training objective minimizes converged SCF energies variationally. The
training geometries are not a set of fitted reference energy/force labels.

## Download and verify

Large payloads are tracked with Git LFS. Install Git LFS before downloading the
data; a pointer-only checkout is not a complete archive. To inspect the code
first and then explicitly obtain the data:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch gradscf_nnao https://github.com/STOKES-DOT/GradSCF.git
cd GradSCF
git lfs pull --include='reproducibility/nnao/**'
python reproducibility/nnao/verify_source.py
python reproducibility/nnao/verify_data.py --full
```

The full verification hashes all archived data, checks the original ordered
500-system structures, and checks inference vectors against their original
numeric parameter hashes. Checksums validate preservation, not convergence or
scientific accuracy; those statuses belong to the original run records.
Historical `SHA256SUMS` and `FILE_MANIFEST.json` files are kept as source
provenance and can include intentionally omitted optimizer/cache files. The
current released payload identities are `data-manifest.json` and `verify_data.py`.

## Model parameters

`models/canonical-nine.json` points to the nine formal training endpoints.
The wider catalog also includes earlier experiments, diagnostic runs, and the
L1/L2 architecture study. Use the catalog and per-model metadata to distinguish
these roles. Duplicate parameter vectors share a hash-addressed payload.

The selected current training endpoints are not automatically the checkpoints
used by every historical figure. In particular, the paper's update-3351 model
is preserved under `paper/data/computational_examples/2026-10-21/models/`.
For reproducing a figure or benchmark, use its recorded frozen checkpoint/basis
identity rather than replacing it with a later endpoint from the catalog.

Each `parameters.npz` contains only numeric inference arrays. Standard models
use `parameters`; the earliest joint coefficient/exponent format retains both
`theta` and `beta`. No Adam state or stored training gradient is included.
The accompanying JSON preserves architecture, basis/candidate identity,
exponent keys, source version and accepted update/epoch information.
Legacy per-case result containers with `adam_state` were also converted:
optimizer arrays are removed while parameters, step counts and scientific
outputs are preserved, with each transformation and original checksum recorded
in the collection manifest.
The export verifies array equality with the original checkpoint. Model
reconstruction still requires its recorded architecture and exponent
parameterization; a flat vector alone is not a complete basis definition.

For an online/minibatch run, the selected endpoint is the latest accepted
`last` checkpoint; its online epoch average must not be described as the full
dataset energy of that parameter vector. For a full-batch run, selection prefers
`best_validation`, then `best`, then `last` where available. Restart/retired
snapshots are recorded in `selection-history.json` rather than promoted to
final inference models. The exports are inference-only and cannot resume Adam.

## Result versions and scientific meaning

- Preserve each run's method, basis, geometry, charge/spin, convergence/stability
  flags, dtype, source identity, backend and units. Failed or incomplete cases
  remain explicitly represented; no missing point is synthesized.
- The 18 curated paper cases include MACE versus free contraction, ethane
  torsion/C–C stretching, s/p/d radial functions and coefficients, ethylene
  density planes, radial moments, polarizability, BSSE, dimers and barriers.
- Prefer the corrected 2026-10-22 ethane stretch and corrected constant-isolated-
  monomer dimer definitions documented in the case index. Superseded versions
  remain in their original dated directories for traceability.
- The historical non-QZ partial accuracy figures remain under Case 18. The
  later complete saved server analysis is a separate 54-row basis/method table;
  its verification requires all 30 structures and all 12 relative observables.
  That campaign excludes CP/BSSE. Earlier dedicated BSSE experiments remain
  separate scientific tests.
- Database relative-energy references are ISO34/W1-F12 and
  S22B/estimated CCSD(T)/CBS. Same-method large-basis energy proxies are a
  different baseline and must not be labelled as database absolute energies.
  Read each protocol's reference-level file before combining results.
- Rebuildable S/h/ERI/DF/JIT caches, compiled libraries, full optimizer
  checkpoints and nonaccepted failure parameter vectors are outside the public
  data payload. Failure diagnostics are retained as metadata and logs; the
  original checkpoints and source workspaces were not modified.

## Sources and attribution

See [DATA_SOURCES.md](DATA_SOURCES.md). Third-party structures/reference values
retain their original provenance and terms. Code licensing must not be treated
as a blanket relicensing of external benchmark data. This preparation does not
create a new dataset DOI or replace the original benchmark citations.
