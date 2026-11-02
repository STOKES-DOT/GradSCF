# GradSCF NNAO training and benchmark workspace

This `gradscf_nnao` branch preserves the GradSCF version used for the NNAO
training campaigns and comparisons through 2026-10-31. It starts from commit
`2940d359f28f51b8efa26e78a3984d79f9585526`, identified in the original campaign
record, and restores the subsequent training and SCF stability fixes from the
actual remote source snapshot. It is a research archive, separate from ongoing
development on `release/v1.0.0`.

## Source and model layout

- `src/gradscf/model/nnao/`: MACE coefficient heads, element/basis definitions,
  model-specific integrals, initialization, training, and workflows.
- `src/gradscf/model/nnao/vendor/mace-jax/`: pinned upstream MACE-JAX source and
  license, retained unchanged inside NNAO.
- `src/gradscf/integrals/_native/`: the matching C/C++ integral backend, pinned
  vendor sources, derivative kernels, and offline build entry point.
- `tests/nnao/` and `tests/comparisons/nnao_*/`: model regressions and comparison
  utilities, including fixed/adaptive dimer protocols and basis-cost accounting.
- `reproducibility/nnao/`: [paper-data index](reproducibility/nnao/README.md),
  training structures, scientific test outputs, figure scripts, inference-only
  weights, campaign recipes, fixed S/M/L exponent pools and source/data hashes.
  Large payloads use Git LFS. Integral caches, binaries and full optimizer
  checkpoints remain in their data workspaces.

Training covers the nine S/M/L × SZP/DZP/TZP families. Exponents are shared by
element/angular channel; MACE predicts complete environment-dependent radial
contraction coefficients. Core basis parameters remain fixed while the SCF
calculation includes all electrons. Fixed-exponent training reuses primitive
S/h/packed-DF caches and differentiates through contraction and implicit SCF.
The separate exponent-derivative and fine-tuning infrastructure is preserved.

## Installation and checks

Use Python 3.11+ and float64. The original GPU training stack used JAX/JAXlib
0.10.0, Flax 0.12.7, and e3nn-jax 0.21.0. The vendored MACE-JAX project declares
its additional dependencies. Install those dependencies in an appropriate
environment before constructing a model; the base package alone is insufficient.

```bash
python -m pip install -e '.[dev]'
python -m pip install -e src/gradscf/model/nnao/vendor/mace-jax
export PYTHONPATH="$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax"
export JAX_ENABLE_X64=1
python -m gradscf.integrals._native.build --jobs 2
python reproducibility/nnao/verify_source.py
python -m pytest -q tests/nnao/test_single_pass_training.py \
  tests/nnao/test_minibatch_training.py tests/nnao/test_energy_stopping.py
```

See `reproducibility/nnao/VERIFICATION.md` for the checks run on this archive.
No new GPU training or full repository regression is implied by that report.

## Campaign recipes and data

`reproducibility/nnao/family500/` records the 500-system campaign configuration
and historical orchestration scripts. `pools/` contains the actual S/M/L
candidate pools. `hidden_l50/` preserves the L=1/L=2 experiment settings and its
completed 300-epoch energy/time summaries. `basis_accuracy/` preserves the
GPU4PySCF MP2/ωB97M-V comparison scripts and protocol, including reference levels;
the active accuracy campaign omits CP/BSSE.

These orchestration files retain historical working-directory paths and input
hashes. Treat them as execution records: use the released reference geometry
JSON, adapt the paths and rebuild a matching native cache before a new run. The model's
portable workflow entry point is
`python -m gradscf.model.nnao.workflows.fixed_pool --help`.
Do not launch the archived scheduler/controller scripts on another cluster
without adapting its paths and resource settings.

The complete ordered 500-system geometry archive is now included at
`reproducibility/nnao/paper/data/training/nnao500-20261021/`. All located model
exports are indexed at `reproducibility/nnao/models/catalog.csv`; the nine
formal training endpoints and their frozen benchmark matches have separate
JSON indexes. The original paper model update 3351 is preserved independently.
Exports contain exact numeric inference parameters and model identities, without
Adam or training gradients; they are not optimizer-resume checkpoints. Original
training checkpoints and source data were preserved unchanged on the server.

## Provenance

`reproducibility/nnao/source-manifest.json` verifies 1,062 scientific/package
files against the actual remote training workspace. Its original runtime
source hash is
`7e5a22c4a17e3d727c03c121be4484e902080cef7b39b164783ce5fde23cdec2`.
That historical hash also included generated CMake JSON. The source-only
scientific payload hash, excluding generated build metadata, is
`3a4fe5813f14b563642d1e1b0f07351d2ddc1c5e8c4b87e217cf05b749eb8ed7`.
This difference is recorded explicitly; it is not a change to scientific code.
Legacy checkpoint/cache guards must not be bypassed when paths, source hashes,
or native libraries change.
