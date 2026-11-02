# Archive verification — 2026-11-02

Source provenance: `python reproducibility/nnao/verify_source.py` verified all
1,062 archived package/scientific files against the actual remote training
snapshot. Native vendor sources passed the build entry point's checksum checks.

Local environment: macOS arm64, Python 3.12.2, JAX/JAXlib 0.8.1, NumPy 2.3.4,
Flax 0.12.1, Optax 0.2.6, pytest 8.3.4. Exact metadata is in
`environment-local.json`. This is distinct from the original GPU training stack.

## Fresh checks

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 \
  python -m gradscf.integrals._native.build --jobs 2
```

Passed: offline C/C++ source build, including coordinate, coefficient, exponent,
packed-DF, and direct-JK native kernels. The generated library/build directories
are ignored and are not part of the Git archive.

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 python -m pytest -q \
  tests/nnao/test_reference_631g.py \
  tests/nnao/test_minibatch_training.py \
  tests/nnao/test_single_pass_training.py \
  tests/nnao/test_energy_stopping.py \
  tests/nnao/test_pool_selection_checks.py \
  tests/nnao/test_basis_accuracy_protocol.py \
  tests/nnao/test_dimer_controls.py
```

**61 passed in 11.10 s.** Covers reference cores, coefficient projection/native
agreement, cache identity, Adam resume/rollback, single-pass evaluation counts,
energy stopping, benchmark bookkeeping, and fixed/adaptive dimer definitions.

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 python -m pytest -q \
  tests/nnao/test_basis_family_grid.py \
  -k 'grid_dimensions or large_single_zeta or grid_workflow'
```

**11 passed, 10 deselected in 14.07 s.** Checks the nine basis-family dimensions
and metadata. Deselected tests include MACE and additional native/SCF cases.

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 \
  python -m pytest -q tests/nnao/test_kr2_stability.py
```

**1 passed in 11.28 s.** Checks the repaired Kr2 fused stability and implicit
coefficient gradient path. Total for these focused checks: **73 passed**.

`python -m pip wheel . --no-deps --no-build-isolation` also passed. The resulting
wheel was checked for the BSE reference data, pool pilot, native CMake file,
and coefficient derivative source. Wheels and compiled libraries are not committed.
Python AST parsing passed for all 79 NNAO test/recipe files; staged-file checks
exclude raw geometry datasets, large checkpoints, binaries, and local settings.

The first check attempt preceded the native build and the archived controller
path adjustment: 58 passed and 3 failed due to those missing runtime/file
dependencies. The complete 61-test command above was rerun after preparation.
No scientific source was changed to obtain these passes.

## Scope

The full repository suite, GPU MACE construction/training, and GPU4PySCF MP2/DFT
campaign were not rerun for this source archive. The earlier A100 L1/L2 runs
completed 300 epochs each; their measurements are preserved under
`hidden_l50/results-20261031/` and are historical results, not new validation.

## Archival completeness review

A separate read-only review verified source/index identity and identified three
relocation/dependency gaps. The benchmark preparation script now resolves the
committed comparison package at the correct repository depth; the original
runtime requirements/environment helpers and family500 submission scripts are
included. Direct import of the relocated preparation script passed, followed by
**12 benchmark/dimer protocol tests passing in 0.03 s**. These are a rerun of
tests already counted above, not 12 additional unique tests.
