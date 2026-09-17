# Publication validation — 2026-09-13

Source commit: `815cc45` (standalone SCF and periodic workflows).

Fresh checks ran from the isolated publication checkout on macOS arm64,
Python 3.12.2, JAX 0.8.1, CPU float64, PySCF 2.9.0. `jax_xc` is not installed
in this local environment. These checks are not a full-suite or GPU result.

## Focused Python regressions

The following invocation passed 67 tests, skipped 1 (the PySCF 2.9 periodic
velocity reference lacks the required API), and deselected 16 numerical
response comparisons, in 25.45 seconds:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_no_external_chemistry_dependency.py tests/test_package_data.py \
  tests/test_integral_namespace.py tests/test_gradscf_package.py \
  tests/test_scf_convergence_policy.py tests/test_scf_diis_state.py \
  tests/test_scf_energy_assembly.py tests/test_scf_xc_protocol.py \
  tests/test_scf_core.py tests/pbc/test_foundation.py \
  tests/pbc/test_kpoints.py tests/pbc/test_optics.py tests/pbc/test_tdscf.py \
  -k 'not periodic_dft_excitations and not periodic_hf_excitations and not unrestricted_gamma_response'
```

The new cached-response regressions failed before the fix and passed after it.
They cover geometry rebuilds and FFT mesh changes, rejection through both
`kernel()` and `get_ab()`, and recovery after rerunning SCF.

## Native build and regressions

A clean offline C/C++ build succeeded on macOS with Clang 16.0.3:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python -m gradscf.integrals._native.build
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
  tests/integrals/test_native_values.py \
  tests/integrals/test_native_geometry_ad.py tests/test_uhf_stability.py
```

The second command passed 43 tests in 16.49 seconds. Combined fresh targeted
checks: 110 passed, 1 skipped; no full-suite pass is claimed.

## Packaging and previous numerical evidence

`python -m pip wheel --no-deps --no-build-isolation .` succeeded. The wheel
contains native C/C++ sources, basis/GTH data, and upstream notices; it contains
no local shared libraries, bytecode, or retired `td_graddft` namespace.
`git diff --cached --check` passed. The existing AGENTS.md was not changed.

The adjacent band and spectrum reports preserve independently run remote
PySCF 2.13 comparisons and their source hashes. Those calculations predate
this cache-validation-only fix; they have not been rerun at commit 815cc45.
The broader ten-system molecular matrix still has documented unconverged
open-shell cases; publication does not imply every method/system converges.
