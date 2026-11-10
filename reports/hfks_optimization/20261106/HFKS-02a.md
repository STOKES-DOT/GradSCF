# HFKS-02a — selected-product density screening

Status: narrow TDD validation passed; end-to-end large-system performance pending.
Baseline: 58b01bb3. No SCF thresholds, basis, grids or AD guards were relaxed.

## Change

The existing shell L1 reference bounds previously took the union of all six
within-/cross-pair density blocks even for a single requested operator.
J-only now uses the two within-pair blocks; K-only uses the four cross-pair
blocks. Joint J+K keeps the original union. Both sets preserve the eight ERI
permutations. The factor 8 and basis Schwarz test remain unchanged.

This intentionally changes the retained subset for positive-cutoff selected
products with an explicit screen reference. It is not an exact identity to the
old union mask for arbitrary payload/reference pairs. Independent dense mask
oracles, fixed-reference linearity and derivative tests validate the new rule.
The default no-reference and zero-cutoff behavior are unchanged. Local screening
bounds are not global error certificates.

## RED and GREEN

- RED (job 116507): four intended assertion failures, no test errors. Each
  J-only/K-only/joint action evaluated 32 intors in the four-s-shell fixture.
- An attempted incremental GREEN build reused the old library because of a
  roughly 139-second filesystem clock skew. The identical library hashes
  exposed this; its results were rejected. No numerical gate was loosened.
- Fresh-directory GREEN (job 116513): 88 CPU tests passed. Six CUDA cases were
  deliberately skipped in that CPU phase, then all six passed on the allocated
  5090 in a separate GPU phase (value, JVP, VJP; Coulomb and long range).
- Consumers: 27 passed, including RHF/UHF/GHF/RKS/UKS/GKS direct/dense parity,
  UKS execution and MGGA derivative regressions.
- Total accepted tests: 121. Raw XML/logs remain in raw/ and are not committed.

| Requested products | RED intor calls | GREEN intor calls |
|---|---:|---:|
| J+K | 32 | 32 |
| J only | 32 | 21 |
| K only | 32 | 29 |

Counts include the identical Schwarz preparation calls. The fixture has a
sparse nonsymmetric density: discarded contributions are exactly zero, and
selected outputs agree with unscreened dense ERIs at atol/rtol 1e-12. These
counts are NOT a measured speedup for the 96-atom molecule.

## Reproduction

Use a fresh native build directory, `python -m gradscf.integrals._native.build
--diagnostics`, and verify the resulting library hash before testing.
CPU: `GRADSCF_EXPECT_DIAGNOSTICS=1 JAX_PLATFORMS=cpu python -m pytest -q
 tests/integrals/test_direct_selected_screen.py
 tests/integrals/test_direct_density_screen.py
 tests/integrals/test_direct_long_range_jk.py
 tests/integrals/test_direct_geometry_response.py`.
GPU: `GRADSCF_TEST_GPU=1 JAX_PLATFORMS=cuda,cpu python -m pytest -q
 tests/integrals/test_direct_selected_screen.py -k cuda`.
Use the single allocated GPU/CPU64, not the login node. See checkpoint.json for
source and native library hashes.

## Still open

HFKS-02d: full six-method/family and large-system numerical/performance gates.
HFKS-03: AO materialization, XC contractions and memory policy.
HFKS-04: preparation lifetimes, training and final combined validation.
Real neural training, all functional categories and a 96-atom speedup have not
been certified by this small first patch.
