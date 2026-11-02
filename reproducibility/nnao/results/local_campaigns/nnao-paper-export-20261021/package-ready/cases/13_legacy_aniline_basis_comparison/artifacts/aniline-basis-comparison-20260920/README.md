# Aniline basis comparison — preflight and validated small-basis results

Prepared on 2026-09-20 against the current GradSCF module layout. This report is not the completed eleven-basis or NNAO training result.

## Geometry and Hamiltonian

Aniline C6H7N, 14 atoms, 50 electrons, neutral singlet, 25 occupied spatial orbitals. All calculations use the same saved geometry, RHF, spherical AOs and float64. The fixed geometry was generated using RDKit ETKDGv3 (seed 20260916) followed by MMFF94 minimization; it is not a quantum-optimized geometry. Original source SHA256: `67e2f01fd1f447ba37676108ac0528ff66003195978aaa5287cf310963e8a23f`.

## Current module layout

- NNAO: `gradscf.model.nnao` (`src/gradscf/model/nnao/`).
- Integrals: `gradscf.integrals`.
- Vendored MACE runtime root: `src/gradscf/model/nnao` for importing `mace_jax`.
- The fresh remote snapshot is `/tmp/gradscf-aniline-20260920`; it does not reuse the former top-level `src/nnao` package. Native source hashes match the previously validated compiled Linux library.

## Sizes and completed energies

| Basis | Contracted AOs | Primitive AOs | Fock matrix | Contracted ERI / GiB | Primitive ERI / GiB | RHF energy / Ha |
|---|---:|---:|---:|---:|---:|---:|
| sto-3g | 42 | 126 | 42 x 42 | 0.023 | 1.878 | -282.208853098886 |
| 3-21g | 77 | 126 | 77 x 77 | 0.262 | 1.878 | -284.137222113526 |
| 6-31g | 77 | 182 | 77 x 77 | 0.262 | 8.175 | -285.626081277790 |
| 6-31g(d) | 112 | 217 | 112 x 112 | 1.172 | 16.521 | -285.726964353351 |
| 6-31g(d,p) | 133 | 238 | 133 x 133 | 2.331 | 23.906 | -285.744255897816 |
| def2-svp | 133 | 217 | 133 x 133 | 2.331 | 16.521 | -285.524681258756 |
| cc-pvdz | 133 | 231 | 133 x 133 | 2.331 | 21.215 | -285.756510140940 |
| def2-tzvp | 259 | 378 | 259 x 259 | 33.527 | 152.110 | Not run |
| cc-pvtz | 308 | 406 | 308 x 308 | 67.049 | 202.439 | Not run |
| NNAO-core6-4s4p2d | 98 | 266 | 98 x 98 | 0.687 | 37.301 | Not run |
| NNAO-core6-6s6p3d-H6s2p | 98 | 399 | 98 x 98 | 0.687 | 188.835 | Not run |

Tensor storage is calculated as 8*N^4 bytes and is not the process peak. Standard RHF uses contracted ERIs. The existing coefficient-training path caches primitive ERIs and differentiates their JAX contraction, requiring much more memory. The current spectral factorization also starts from a full ERI matrix and does not remove the allocation.

## Verification

Local focused regression: 36 passed, 2 optional MACE dependency skips; the self-contained molecular tests were then rerun (4 passed) after removing their artifact dependency. The final molecular/dependency checks had 6 passed, 1 optional skip. The new tests cover C/H/N model assembly, a nitrogen-containing molecule's energy gradient, and pre-allocation guards. Remote outcomes are in remote-tests.log. Remote focused regression: 33 passed in 67.12 seconds, including the current C/H/N MACE assembly. SCF results and independent PySCF checks are in remote-smoke/, remote-polarized/ and their reference logs.

## Full run awaiting a resource decision

The large-memory job was launched on c20 with 8 CPU cores and a 700 GiB virtual-memory limit. The seven small/medium standard bases completed; def2-TZVP failed after 15:59.59 wall time with an SCF residual of 157.64, even before the level-shift retry. The retry used damping=0.35, level_shift=0.5 and 500 cycles, ran 22:?? wall time, and still failed with residual 136.57. cc-pVTZ was therefore not started. The NNAO core6 + 4s4p2d process allocated about 69.5 GiB RSS and reached a converged SCF (27 cycles, orbital residual 7.76e-13), but its primitive-to-contracted energy reconstruction error was 40.0706 Ha, so the safety check stopped before optimization. The 6s6p3d run was not started.

The failure is now narrowed: memory allocation succeeded, but the large-basis GradSCF SCF path is unstable for def2-TZVP and the primitive ERI/contraction energy consistency check fails for the aniline NNAO layout. These rows are excluded from the numerical comparison table until fixed.

The original `run_full.sh` remains as a reproducibility script, but it should not be rerun unchanged until these two issues are addressed.

The high-memory job requires user approval under the user-provided Global AGENTS.md instruction to ask before long remote jobs that consume noticeable resources. An alternative is to implement and validate a blockwise/low-memory integral path first. No methane-trained coefficients or energies are presented as aniline results.
