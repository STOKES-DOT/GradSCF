# Extended molecular EOM forward study

2026-09-23. Production baseline `73f27a4`; no production SCF, CC or eigensolver
code changed in this study. CPU (`TFRT_CPU_0`), macOS arm64, Python 3.12.2,
JAX 0.8.1, PySCF 2.9.0, float64. OMP/OpenBLAS/MKL thread variables each set to 1.
Geometries are fixed numerical fixtures in Angstrom, not optimized structures
or experimental spectroscopy predictions.

## Controls and initial sweep

H2 at 1.5 and 2.5 A, He2 at 3.0 A, Ne, linear CO2/HCN/C2H2, F2 at 1.42 and
2.2 A, and bent O3. H2/He2/Ne use 6-31G; the others STO-3G. Exact coordinates
are in `tests/comparisons/compare_eom_stress.py`. A later frozen-space C4H6
fixture adds a larger pi-system geometry.

Each sector requests three lowest roots: Davidson space=48, max_cycle=180,
guard_roots=1, seed=0, residual tolerance 1e-9 Ha. Independent full PySCF action
matrices establish the reference ordering. Acceptance is 1e-8 Ha plus successful
left/right/guard checks. Native CC energy/residual tolerances are 1e-12/1e-11 Ha,
max_cycle=200. Initial reference CC energy/update-norm criteria are 1e-13/1e-12,
max_cycle=200. Native HF uses hcore, energy tolerance 1e-12 and default
density/gradient criteria. No automatic retries occur in the first sweep.

| Fixture | Basis | Initial outcome | Maximum spectral error / Ha | Seconds including SCF/compile |
| --- | --- | --- | ---: | ---: |
| H2_1.5 | 6-31g | 3/3 passed | 1.285e-12 | 26.50 |
| H2_2.5 | 6-31g | 3/3 passed | 1.521e-10 | 27.45 |
| He2 | 6-31g | 3/3 passed | 3.443e-09 | 21.44 |
| Ne | 6-31g | 3/3 passed | 8.606e-13 | 28.84 |
| CO2 | sto-3g | 3/3 passed | 5.213e-11 | 32.64 |
| HCN | sto-3g | 3/3 passed | 2.067e-11 | 29.45 |
| C2H2 | sto-3g | 3/3 passed | 3.695e-12 | 30.32 |
| F2_1.42 | sto-3g | 3/3 passed | 1.391e-11 | 29.39 |
| F2_2.2 | sto-3g | Reference CC not converged; EOM withheld | — | 5.91 |
| O3 | sto-3g | Reference CC not converged; EOM withheld | — | 6.68 |

The first sweep passed **8/10 fixtures and 24/30 planned sectors**. Per-case
timers sum to 238.61 s including compilation and reference
calculations; these are reproduction timings, not performance claims. Both
native CC calculations in the withheld cases had converged. The withheld cases
must not be mislabeled as native EOM failures.

## Findings and controlled followups

### Weak-interaction satellites depend on SCF precision

He2 passed the 1e-8 Ha threshold, but its third IP/EA roots differed by 3.44e-9
and 2.56e-9 Ha despite HF/CC total-energy agreement near 1e-14 Ha. EOM residuals
were around 1e-15 and full spaces were solved; increasing Davidson effort would
not resolve this discrepancy. Tightening only native HF energy/density/gradient
tolerances to 1e-14/1e-12/1e-11 (max_cycle=150) reduced maximum IP/EA differences
to 8.77e-12 and 4.53e-12 Ha. Total-energy agreement alone does not certify orbital
or satellite-energy accuracy. This is a precision-control issue in an already
passing case, not evidence of a wrong EOM contraction.

### O3 needed a longer independent CC reference

Native CC initially converged with residual 5.47e-12 Ha. The reference 200-cycle
run did not reach its stricter update-norm criterion: update norm 7.39e-12 and
physical residual 5.42e-12 Ha. A restart from its own amplitudes, max_cycle=600
and DIIS space=8, retained the strict thresholds and converged to update norm
9.60e-13 and residual 7.05e-13 Ha. Native HF density/gradient criteria were also
set to 1e-10/1e-9. All three sectors then passed, maximum error 3.73e-10 Ha.

Reference physical residuals are evaluated with GradSCF's independently validated
CCSD equations in the reference MO frame; they supplement, not replace, the
reference solver's own convergence flag and update norm.

### Stretched F2: finite multistart can miss a known lower branch

At 2.2 A, native hcore reaches -195.524616699278 Ha; the reference HF energy is
-195.671626240736 Ha. The previous single seed with amplitudes (0.05,0.15,0.4)
only reaches -195.596566567657 Ha. Every candidate can converge while the finite
set still misses a known lower branch.

Expanded native search: seeds (0,1,20260923), each with amplitudes (0.25,0.5,1.0),
retaining baselines and all attempts. Seed 1/amplitude 0.5 and seed
20260923/amplitude 1.0 reach the reference energy. The selected native CC energy
is -195.97673225288247 Ha, residual 4.22e-12 Ha. Independent FCI gives
-195.97673225288278 Ha: absolute difference about 3.1e-13 Ha. No PySCF density
or orbitals initialized native SCF.

The optimized reference CC failed after 200 and restarted 600-cycle budgets;
a further trial with update-norm tolerance 1e-10 also remained unconverged.
None was used as a qualified EOM oracle. A separate PySCF RCCSD reference,
retaining the original strict 1e-13/1e-12 criteria, converged after permitting
a further max_cycle=5000 restart: update norm 9.98e-13, residual 5.31e-13 Ha.
These cycle numbers are **budgets**, not measured iteration counts.

Against its converged full reference spectrum, maximum native differences are
2.25e-13 (EE), 2.70e-12 (IP) and 2.27e-13 Ha (EA). FCI singlet excitation and
attachment energies additionally agree with EE/EA in this small one-virtual
fixture. IP is compared to EOM-CCSD, not claimed exact against FCI. The final
native search, FCI and strict reference diagnostic took 34.37 s including
compilation. This is evidence for the selected branch, not a global guarantee
for finite multistart on general systems.

### Frozen-space larger pi-system geometry

C4H6/STO-3G has 26 MOs and 15 occupied orbitals. Freeze occupied [0,1,2,3] and
virtual [19,20,21,22,23,24,25], zero based. This leaves 11 occupied and 4 virtual
orbitals active; EE dimension 1034. Maximum errors are 4.80e-10, 1.73e-10 and
2.61e-10 Ha for EE/IP/EA. It is an explicitly truncated numerical fixture,
not full-space CCSD or a pi-only active-space spectroscopy prediction.

## Final accepted comparisons

| Fixture | Basis | Maximum error / Ha | Additional controls |
| --- | --- | ---: | --- |
| H2_1.5 | 6-31g | 1.285e-12 | Initial settings |
| H2_2.5 | 6-31g | 1.521e-10 | Initial settings |
| He2 | 6-31g | 8.773e-12 | Tighter native SCF |
| Ne | 6-31g | 8.606e-13 | Initial settings |
| CO2 | sto-3g | 5.213e-11 | Initial settings |
| HCN | sto-3g | 2.067e-11 | Initial settings |
| C2H2 | sto-3g | 3.695e-12 | Initial settings |
| F2_1.42 | sto-3g | 1.391e-11 | Initial settings |
| F2_2.2 | sto-3g | 2.698e-12 | Expanded native SCF; converged strict RCCSD reference |
| O3 | sto-3g | 3.728e-10 | Longer reference CC; tighter native SCF |
| C4H6_frozen | sto-3g | 4.800e-10 | Explicit frozen window |

With the documented followups, **11 fixtures, 33 sectors and 99 distinct roots**
have accepted forward comparisons. This is not an 11/11 default-run claim.
[STRESS_VALIDATION.csv](STRESS_VALIDATION.csv) retains 108 root records, including
both original and tightened He2 results. Largest accepted error: 4.80e-10 Ha.
Withheld ground-state attempts are recorded above, without invented excitations.

No additional EOM algebra or real-left-subspace defect was observed. Remaining
issues center on SCF branch coverage, reference convergence budgets and
propagation of SCF precision. Degenerate groups retain invalid isolated-energy
AD flags. No new AD finite differences or nuclear gradients were checked;
partial iterative spectra still do not certify global root ordering. Small
basis-set attachment levels are not claims of experimentally bound anions.

## Reproduction

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/comparisons/compare_eom_stress.py > stress.jsonl

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/comparisons/compare_eom_stress_followups.py > followups.jsonl

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/comparisons/diagnose_stretched_f2.py > f2.jsonl
```

The scripts have no CLI. The comparison helper accepts explicit reference
objects and frozen orbitals for controlled experiments; default calculations
are unchanged. No production solver, dependency, merge or push changed here.
