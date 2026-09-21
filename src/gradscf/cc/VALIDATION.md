# Restricted ground-state CC validation — 2026-09-21

Environment: arm64 CPU (`TFRT_CPU_0`), JAX 0.8.1, PySCF 2.9.0, float64.
Tests were run in the isolated `feat/cc-ground-state` worktree.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
python -m pytest -q tests/cc tests/ci tests/solvers \
  tests/test_scf_diis_state.py tests/test_scf_diis_scaling.py \
  tests/test_tddft_eigensolvers.py tests/test_scf_higher_order.py
```

Initial comprehensive result: **129 passed**, two PySCF OpenMP-availability warnings, 185.64 seconds.
No tests in this selection were skipped. This is targeted validation, not the
complete repository test suite or a GPU benchmark.

The 34 CC tests in that run include:

- CCSD/CC2 random-amplitude residuals versus PySCF;
- independent determinant-space BCH checks for CCSD, CCD and CCS, and
  H + [H,T] checks for LCCD/LCCSD;
- energies, right amplitudes and spin-adapted Lambda amplitudes;
- conventional (T) on H4 and H2O, and the two-electron zero-(T) limit;
- frozen core/virtual spaces and the fully frozen reference limit;
- two-electron FCI and noninteracting-fragment additivity;
- implicit energy response for every exported model, amplitude response,
  CCSD(T) total response, and independence from converged DIIS trajectories;
- nonconvergence, invalid settings, unsupported methods and complex amplitudes.

A subsequent float32-restart/float64-integral regression exposed a JAX loop dtype
mismatch. Restart amplitudes now use the common integral/amplitude dtype before
packing. The new regression, namespace smoke test and nonconvergence derivative
test passed together: **3 passed**, 12.56 seconds (`tests/cc -k
'lower_precision or cc_public or failure_and_unsupported'`). The current CC test
file therefore contains 35 cases. The wheel was rebuilt after this code fix.

## Runnable H4 example

`examples/cc/restricted_ground.py` uses H4 at z = 0, 0.8, 1.9, 3.1 Angstrom,
STO-3G, real RHF and the native CPU integral path. Energies are Hartree.

| Method | Total energy |
| --- | ---: |
| RHF | -2.089978546810821 |
| CCS | -2.089978546810820 |
| CCD | -2.161519207807380 |
| LCCD | -2.165569628785267 |
| LCCSD | -2.165593627670407 |
| CC2 | -2.132771236216811 |
| CCSD | -2.161533025614623 |
| CCSD(T) | -2.161619299668401 |

These methods are not variational bounds; a lower number in this table does not
establish a more accurate method. CCSD converged in 12 iterations with residual
3.20e-11 Hartree; the Lambda linear residual was 1.71e-16. The (T) correction
was -8.627405377840567e-5 Hartree.

For a dimensionless coupling scale that preserves the canonical Fock matrix,
the complete CCSD(T) derivative was -2.5222909909633544 Hartree. Central finite
difference with step 1e-4 gave -2.5222909920175063 Hartree, absolute difference
1.054e-9 Hartree. This is an integral-parameter test, not a nuclear-force test.

## Packaging and runtime independence

- A no-index, no-dependency wheel build succeeded.
- The wheel includes the CC implementation, shared helpers, Apache license,
  source-code NOTICE and bibliographic files.
- A separate process blocked every PySCF import and successfully ran CCSD,
  Lambda and (T) on a two-orbital noninteracting Hamiltonian.
- `git diff --check` and local documentation-link checks passed.

See [README.md](README.md) for limits and [REFERENCES.md](REFERENCES.md) for
the distinction between theory, adapted code and independent numerical oracles.

## OpenMolcas-informed refinements

The same comprehensive command above, after adding the refinements in
`tests/cc/test_refinements.py`, completed with **142 passed**, two PySCF
OpenMP-availability warnings, no skips, in 233.95 seconds (CPU/float64).
The CC selection now includes 47 cases. The 12 new cases cover:

- iteration-level-shift invariance of energy and implicit response;
- explicit Urban and conventional triples variants and diagnostic components;
- failed/shape-inconsistent/mismatched states, including empty triples spaces;
- full/frozen-space 1-RDM comparisons with PySCF;
- density response including both right and left-state response;
- invalidation after changing the reference, method or frozen space;
- an independent determinant-space WT2 moment check and a finite-difference
  derivative of the Urban correction itself.

The updated H4 example gave Urban's correction -8.642093119764708e-5 Hartree,
the conventional correction -8.627405376957714e-5 Hartree, and the added singles
component +1.4687742806995303e-7 Hartree. The minimum triples denominator magnitude
was 2.0090627405364425 Hartree. The 1-RDM trace was 4.0 and its symmetry error
was zero. The complete conventional CCSD(T) response still agreed with central
finite difference to 1.0542e-9 Hartree along the documented canonical path.

The inspected OpenMolcas source revision and hashes are recorded in
[OPENMOLCAS.md](OPENMOLCAS.md). These checks do not claim that an OpenMolcas
executable was run or that its ROHF/Cholesky implementations were ported.
