# Methane: expanded-basis comparison in GradSCF

All energies were recomputed with GradSCF native integrals and its RHF SCF solver. PySCF was used only as an independent test reference.

## Conditions

Tetrahedral CH4, C-H = 1.09 Angstrom; charge/spin = 0/0; 10 electrons; spherical AOs; float64. No ECP. CPU: Intel(R) Xeon(R) Platinum 8457C; affinity [0, 1, 2, 3]; JAX 0.10.0. Original ten-basis GradSCF elapsed: 56.958 s (includes compilation; not a steady-state speed benchmark).

RHF settings: max_cycle=200, energy tolerance 1e-12 Ha, density tolerance 1e-10, orbital tolerance 1e-9, hcore initial guess. The original ten rows are retained from the preceding verified comparison. The added szp663_direct row is a new GradSCF calculation after MACE optimization. Both NNAO coefficient sets were optimized on this exact methane geometry; these results do not establish transferability.

## Results

| Basis | C AOs | AOs per H | Total AOs | Primitive AOs | Fock matrix | Energy / Ha | Above cc-pVTZ / mHa | SCF cycles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| STO-3G | 5 | 1 | 9 | 27 | 9 x 9 | -39.726700052055 | 486.600920 | 8 |
| 3-21G | 9 | 2 | 17 | 27 | 17 x 17 | -39.976739994771 | 236.560977 | 13 |
| 6-31G | 9 | 2 | 17 | 38 | 17 x 17 | -40.180384752149 | 32.916220 | 13 |
| 6-31G(d) | 14 | 2 | 22 | 43 | 22 x 22 | -40.194732277383 | 18.568695 | 13 |
| NNAO 4s4p2d / H3s1p | 10 | 4 | 26 | 56 | 26 x 26 | -40.172528224491 | 40.772747 | 11 |
| NNAO 6s6p3d / H6s2p | 10 | 4 | 26 | 93 | 26 x 26 | -40.186970229395 | 26.330743 | 11 |
| 6-31G(d,p) | 14 | 5 | 34 | 55 | 34 x 34 | -40.201566245053 | 11.734727 | 13 |
| def2-SVP | 14 | 5 | 34 | 52 | 34 x 34 | -40.169177389196 | 44.123583 | 13 |
| cc-pVDZ | 14 | 5 | 34 | 54 | 34 x 34 | -40.198710292819 | 14.590679 | 13 |
| def2-TZVP | 31 | 6 | 55 | 78 | 55 x 55 | -40.212963597132 | 0.337375 | 13 |
| cc-pVTZ | 30 | 14 | 86 | 106 | 86 x 86 | -40.213300971922 | 0.000000 | 13 |

## Matrix interpretation

The SCF eigenproblem is F C = S C epsilon. F is the effective one-electron Fock matrix; Hcore and S have the same N x N shape. GradSCF forms an N x N orthogonalized Fock matrix for diagonalization. N is the contracted AO count, including angular components, not the primitive count. Methane has 5 occupied spatial orbitals for every basis; all N orbitals are retained by this solver. No overlap eigenvalues in this experiment approach the clipping threshold.

## Verification and reproducibility

All eleven SCFs converged; every Fock and MO matrix shape was checked, as were FDS-SDF and generalized eigen-equation residuals. Independent PySCF 2.13.0 agrees in AO count for every row; maximum energy difference = 1.0445e-12 Ha (acceptance tolerance 1e-8 Ha).

A preflight detected and fixed a basis-name parsing defect: heavy-atom polarization in single-part parentheses, e.g. 6-31G(d), was previously ignored. The fix is covered by star-alias equivalence and multi-polarization regression tests. All results here use the corrected parser.

On a fresh copy of /tmp/gradscf-core6-comparison-20260916 with the recorded code overlay, run gradscf-szp663-check.sh, then gradscf-szp663-refine.sh in /tmp/gradscf-szp663-20260916. The first script records the 160-step trial; the second repeats from the same initialization with maxiter=600 and reaches gradient convergence at step 463. The original ten-row benchmark remains in ../methane-core6-basis-comparison-20260916. Source hashes and raw results for the new row are retained under artifacts/methane-szp663/. See optimization-details.md for initialization, optimizer convergence and derivative checks. append_result.py merges the recorded results; render_report.py produces PNG/PDF figures.
