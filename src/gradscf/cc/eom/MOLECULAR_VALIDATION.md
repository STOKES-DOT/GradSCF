# Eight-molecule EOM-CCSD validation

2026-09-23. Code baseline: `ac93b91`. No production SCF/CC/EOM or numerical
solver code was changed during this study. This is a numerical forward study,
not a spectroscopic accuracy benchmark or a new nuclear-gradient validation.

## Setup

macOS arm64 CPU (`TFRT_CPU_0`), Python 3.12.2, JAX 0.8.1, PySCF 2.9.0,
float64; OMP, OpenBLAS and MKL thread environment variables each set to 1.
Each package independently runs restricted HF and CCSD. GradSCF uses its native
integrals and SCF with the default hcore guess; PySCF uses its own default guess.
All geometries are fixed fixtures in Angstrom, recorded in the script, rather
than optimized structures. No frozen orbitals. Native HF energy tolerance
1e-12 Ha; CC energy/residual tolerances 1e-12/1e-11 Ha, max_cycle=200. PySCF HF
and CC energy tolerance 1e-13 Ha and CC amplitude tolerance 1e-12.

All three sectors request the lowest three roots, `solver="davidson"`,
`max_space=48`, `max_cycle=180`, `guard_roots=1`, `seed=0`, residual tolerance
1e-9 Ha. PySCF **full action matrices** are diagonalized independently, avoiding
an assumption that an iterative driver's default guesses find every lower root.
The acceptance threshold is 1e-8 Ha together with all forward convergence
checks. A deliberately invalid isolated-energy response at a degeneracy is
not classified as a forward failure.

## Default-setting results

| Molecule | Basis | EE / IP / EA dimensions | Largest energy difference / Ha | Forward sectors passing | Seconds |
| --- | --- | --- | ---: | ---: | ---: |
| LiH | 6-31g | 189 / 38 / 171 | 8.705e-12 | 3/3 | 29.59 |
| HF | 6-31g | 495 / 155 / 186 | 7.462e-11 | 3/3 | 29.00 |
| N2 | sto-3g | 252 / 154 / 66 | 1.005e+00 | 0/3 | 30.13 |
| CO | sto-3g | 252 / 154 / 66 | 1.285e-10 | 2/3 | 29.87 |
| NH3 | sto-3g | 135 / 80 / 48 | 1.272e-10 | 3/3 | 28.06 |
| CH4 | sto-3g | 230 / 105 / 84 | 4.024e-12 | 3/3 | 28.02 |
| CH2O | sto-3g | 560 / 264 / 132 | 1.755e-10 | 3/3 | 28.46 |
| C2H4 | sto-3g | 1224 / 392 / 294 | 6.526e-12 | 3/3 | 29.73 |

The first sweep completed **6/8 molecules and 20/24 sectors** with all forward
checks satisfied. Per-case elapsed times sum to 232.86 s,
including compilation, ground states and dense reference calculations. Caches
were cleared between molecules; these timings are not performance claims.
N2 and CO were retained as failures for diagnosis, not replaced in this table.

Except for the different N2 HF branch, all energies match the reference within
1.8e-10 Ha. CO's energy agreement alone is insufficient because its EE left/right
state check fails. Detailed energies for all 72 original roots and all controlled
followups are in [MOLECULAR_VALIDATION.csv](MOLECULAR_VALIDATION.csv).

## N2: different SCF branch

The default native hcore start converged to HF energy -106.769673857813 Ha,
whereas PySCF found -107.496500511798 Ha. The difference is 0.726826653985 Ha;
the corresponding CCSD total energies differ by 0.708505218253 Ha. Negative EE
roots around that higher-energy reference cannot be compared as excitations
from the reference ground state. A converged CC residual or EOM eigenpair does
not certify that the preceding SCF state is the lowest physical branch.

A controlled native-only restart used the existing `orbital_rotation_guesses`
helper on the native orbitals, seed 20260923, amplitudes 0.05, 0.15 and 0.4.
The initial density was 2*C_occ*C_occ.T. No PySCF orbitals or density were used.
Restarted SCF used energy/density/gradient tolerances 1e-12/1e-10/1e-9 and
max_cycle=150. The lowest converged candidate was selected explicitly:

| Rotation amplitude | HF energy / Ha | Converged |
| --- | ---: | --- |
| 0.00 | -106.769673857813 | True |
| 0.05 | -106.769673857813 | True |
| 0.15 | -107.496500511798 | True |
| 0.40 | -107.496500511798 | True |

Both 0.15 and 0.4 reached the reference energy. On the selected native branch,
all nine EE/IP/EA roots pass. Maximum differences are
5.161e-13, 7.544e-14 and
1.782e-13 Ha respectively. This establishes agreement
for the selected branch, not a general global-minimum guarantee for multistart.
The default initial-guess behavior remains unchanged.

## CO: degenerate left subspace loses rank

The default native HF and CCSD energies agree with PySCF to 2.84e-14 and
2.44e-10 Ha. The first two EE roots are degenerate near 0.33220364255 Ha;
all three EE energies agree to 6.34e-11 Ha. Nevertheless, with space=48/seed=0,
the returned left-vector matrix has smallest singular value 7.04e-17 and
||L.T R-I||=3.055. Right vectors remain independent (smallest singular value
0.806). Thus this is a genuine failure of the forward left/right state output,
correctly rejected by its convergence flag, even though the eigenvalues agree.

Controlled EE followups reuse the same ground state and retain every attempt:

| Solver | Space | Seed | Residual tolerance | Forward converged | ||L.T R-I|| | Smallest left singular value |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| davidson | 48 | 0 | 1e-09 | False | 3.055e+00 | 7.044e-17 |
| davidson | 48 | 1 | 1e-09 | True | 1.247e-15 | 7.581e-01 |
| davidson | 64 | 0 | 1e-09 | True | 4.286e-16 | 8.305e-01 |
| davidson | 48 | 0 | 1e-10 | False | 2.341e+00 | 1.512e-16 |
| dense | 48 | 0 | 1e-09 | True | 7.143e-16 | 1.004e+00 |

Changing the seed to 1 or enlarging the search space to 64 restores the forward
checks; the dense GradSCF reference also passes. Merely tightening the residual
tolerance to 1e-10 does not fix the default rank loss. These are controlled
workarounds, not a production-code repair. Degenerate left/right subspace
construction remains a robustness target. The observations locate the failure
in the left-space representation/normalization; no claim is made here about
which internal eigendecomposition step is solely responsible.

## Response diagnostics

All three sectors for CH2O and C2H4 have `response_valid=True` for the requested
root group. LiH, HF, NH3 and CH4 include degenerate roots and have False for the
entire group; the corresponding full reference spectra confirm the small gaps.
Correct-branch N2 and the successful CO followups likewise retain False because
of degeneracy. This is the declared isolated-root AD policy, not evidence that
the forward energies failed. No new finite-difference gradient sweep was run;
response flags alone are not measured gradient accuracy.

All iterative spectra remain partial and do not certify global ordering or
absence of unseen roots. The complete independent spectra establish agreement
for these specific fixtures. EA energies mean E(N+1)-E(N); conventional electron
affinities have the opposite sign. STO-3G/6-31G attachment levels are numerical
fixtures and are not claims of experimentally bound anions.

## Reproduction

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/comparisons/compare_eom_molecules.py > sweep.jsonl

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/comparisons/compare_eom_molecular_followups.py > followups.jsonl
```

Both scripts have no CLI. The first emits progress and per-molecule JSON records;
the second records the native N2 restarts and the full CO parameter sweep.
The validation does not change defaults or silently retry failing cases.
