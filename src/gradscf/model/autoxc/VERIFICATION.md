# Infrastructure verification — 2026-11-16

This first section is the original infrastructure-only receipt. The subsequent
live molecular acceptance is recorded below and in the linked example report.

Branch: `feat/xc-agent-design-20261116`, based on `83d43570`.
Implementation remains in this worktree; AutoXC's vendor is unchanged.

Environment: macOS arm64, CPU, Python 3.12.2, JAX/JAXlib 0.8.1, NumPy 2.3.4,
SciPy 1.15.2, float64; Codex Python SDK installed version 0.147.0. Test invocations
set OMP_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=1; these are thread-pool controls,
not a claim of exclusive CPU affinity.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/iks tests/model tests/test_eager_minimize.py tests/test_spherical_representation.py --ignore=tests/iks/test_reference.py
```

Result: **196 passed in 71.41 seconds**, exit 0, no skipped/failed tests.
The molecule-containing `test_reference.py` was explicitly excluded: this phase
uses array/quadrature/interface tests, not local molecular calculations.

New regressions were observed failing before implementation/fixes. They cover
one-XC Hamiltonians, fixed-target energy AD, RKS/UKS/gauge, internal degeneracy
versus a closing boundary gap, scalar custom/neural callbacks, blocked local
potential assembly, strict batch validity, shared bounded fitting, formula
parsing, structured SDK calls, timeout cancellation, duplicate rejection,
saved proposals, failed/interrupted first-turn thread IDs and indirect numerical
source fingerprint changes. Independent read-only review found two Important
restart/fingerprint issues; both received failing regressions and fixes.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python examples/iks/agent_loop.py
```

The live three-round replay exits 0. All three trial powers reach reward
1.52269531 within the displayed precision; the first candidate `a*n**2` remains
incumbent and fits `a=0.1999999999999991`, against the planted value 0.2.
The two-AO sample cannot identify a unique functional family. This is workflow
verification, not molecular accuracy, a PBE improvement, or live agent discovery.
Earlier implementation receipts are preserved in the ignored
`artifacts/iks-agent-loop-history/`; the current run is in
`artifacts/iks-agent-loop-replay/`.

GradSCF Python source fingerprint:
`bf066f8893a0b39b1e61f768d24dc1a3e1ad5a5347297fa77037c9d86b7efcba`.

Not performed: real SDK authentication/model calls, XC100 molecular evaluation,
GPU testing, native-binary reproducibility validation, or sealed-data process
isolation deployment. No dependencies were installed, no remote jobs launched,
and no branch merge/push was performed.

## Subsequent local molecular acceptance

The later scope explicitly authorized local BH3/XC100 numerical work and two
real Codex SDK turns. See [the complete single-molecule receipt](../../../../examples/iks/XC100_CODEX_SINGLE.md).
That run used the original 72-spherical-AO cc-pVTZ convention and level-5 grid,
passed independent PBE/libxc checks, and completed two proposals in one
GPT-5.6 Luna/high thread. Source/data manifests and raw run records remain
outside Git. This supersedes the earlier section's unperformed-live-run status
for that one sample; GPU and broader molecular validation remain unperformed.

The expanded regression command is:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gradscf-mplcache python -m pytest -q tests/iks tests/model tests/test_eager_minimize.py tests/test_spherical_representation.py tests/test_xc_design_pbe.py tests/test_xc_design_grid.py --ignore=tests/iks/test_reference.py
```

It passed 208 tests in the live-acceptance phase. The same bounded pre-merge
suite was rerun successfully: **208 passed in 61.76 s**, exit 0. No additional
model call or molecular re-evaluation was needed for integration.
