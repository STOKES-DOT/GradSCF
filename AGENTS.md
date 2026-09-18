# Repository Guidelines

## Project Structure & Module Organization

GradSCF is a Python/JAX toolkit for Hartree-Fock/DFT, differentiable SCF, response theory, and Neural XC training. The distribution and import namespace are both `gradscf`.

- `src/gradscf/`: public facades (`gto`, `dft`, `tdscf`), SCF kernels, response solvers, molecular data, and XC (`dft.xc` for classic functionals, `dft/libxc_jax` for the JAX/libxc backends). `src/gradscf/gw/`: differentiable GW — molecular G0W0-CD (R/U), evGW, qsGW, scGW, periodic Gamma/k-point KRGW with q->0 head/wing corrections (see module docstrings for staged AD coverage). `src/gradscf/model/`: neural-network models — `neural_xc` (neural XC functionals), `neural_d` (dispersion), `training` (trainers), `nnao` (MACE-conditioned neural basis sets; vendored mace-jax remains a separate top-level package).
- `src/gradscf/integrals/`: canonical integral API, basis parameters, execution plans, and input assembly. `backends/jax_reference/` preserves the JAX reference kernels.
- `native/`: pinned upstream C sources, private C++ FFI, and offline CMake build. Keep vendor files unchanged; record adaptations in patches.
- `src/gradscf_tools/`: reusable workflow helpers; `tools/` contains experiment and evaluation CLIs.
- `tests/`: unit, API-boundary, and numerical regression tests.
- `examples/`: runnable public-API demonstrations.
- `reproducibility/v1.0.0/`: selected checkpoints, reference tables, and figures.
- `src/gradscf/data/pyscf_basis_snapshot/`: bundled basis assets; preserve package-data declarations when changing these files.

## Build, Test, and Development Commands

Run commands from the repository root in a virtual environment. Package support starts at Python 3.10; use Python 3.11+ for tests that import `tomllib`.

- `python -m pip install -e ".[dev,upstreams]"`: install editable sources, pytest, PySCF, and `jax-xc`.
- `python -m pip install -e ".[dev,reproducibility]"`: install manuscript evaluation dependencies.
- `python -m pytest -q tests/test_tddft_eigensolvers.py`: run focused solver tests.
- `python -m pytest -q`: run the complete test suite.
- `python examples/compare_pyscf_vs_jax_tddft_no_neural.py`: compare water B3LYP excitation results with PySCF.

Packaging uses setuptools through `pyproject.toml`.

Build native CPU integrals with `PYTHONPATH=src python -m gradscf._native.build`
before running `tests/integrals/`. Native forward/JIT currently has no AD rule;
never silently replace missing derivatives with zeros or a fallback backend.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Follow surrounding type annotations and dataclass patterns. Prefer public facades in examples and tools. No formatter or linter is configured in `pyproject.toml`; avoid unrelated formatting changes.

## Testing Guidelines

Use pytest with `tests/test_*.py` files and `test_*` functions. Add focused regressions for changed behavior before broad validation. Tests default to CPU and JAX float64 through `tests/conftest.py`. No coverage threshold is configured. Report skipped dependency/backend tests explicitly.

For numerical comparisons, specify molecule, basis, XC, grid, units, convergence criteria, and tolerances. Preserve dtype and seed assumptions; enable float64 before constructing reference arrays.

## Commit & Pull Request Guidelines

History predominantly uses `feat:`, `fix:`, `test:`, `refactor:`, and `docs:` prefixes with concise imperative summaries. Keep commits focused. PRs should describe the behavior change, link relevant issues, list validation commands/results, and disclose unverified backends. For benchmarks, record hardware, elapsed time, and reference errors. Keep raw datasets, caches, and temporary outputs out of commits.
