# Aniline GPU4PySCF basis comparison

All rows use the same fixed aniline geometry, spherical AOs, RHF, float64, and `cc-pVQZ-JKFIT`. The trained NNAO basis is evaluated independently by GPU4PySCF; the training energy is not reused.

| Basis | Level | AO / H dimension | Energy / Ha | vs cc-pVTZ / mHa | SCF cycles | Time / s |
|---|---|---:|---:|---:|---:|---:|
| minao | minimal | 42 | -285.147516390566 | 682.660 | 33 | 1.13 |
| sto-3g | minimal | 42 | -282.208873290139 | 3621.303 | 30 | 1.19 |
| 3-21g | split-valence | 77 | -284.137102782880 | 1693.073 | 33 | 0.92 |
| 6-31g | double-zeta | 77 | -285.626073513251 | 204.103 | 35 | 1.04 |
| 6-31g(d) | polarized double-zeta | 112 | -285.726951800260 | 103.224 | 32 | 1.39 |
| 6-31g(d,p) | polarized double-zeta | 133 | -285.744243985263 | 85.932 | 32 | 1.18 |
| def2-svp | polarized double-zeta | 133 | -285.524685756675 | 305.490 | 34 | 1.01 |
| cc-pvdz | correlation-consistent double-zeta | 133 | -285.756504412903 | 73.672 | 35 | 1.20 |
| 6-311g(d,p) | polarized triple-zeta | 168 | -285.797492280832 | 32.684 | 38 | 1.36 |
| def2-tzvp | polarized triple-zeta | 259 | -285.836082897001 | -5.907 | 39 | 1.73 |
| def2-tzvpp | polarized triple-zeta | 315 | -285.839688214850 | -9.512 | 39 | 2.18 |
| cc-pvtz | correlation-consistent triple-zeta | 308 | -285.830176254292 | 0.000 | 39 | 2.04 |
| NNAO-szp663-trained | trained single-zeta plus polarization | 98 | -285.557554670261 | 272.622 | 39 | 2.88 |

## Interpretation

- The trained NNAO has **98 AOs**, between unpolarized 6-31G/3-21G (77 AOs) and 6-31G(d) (112 AOs).
- Its nearest conventional energy is **def2-svp**: NNAO is 32.869 mHa lower.
- NNAO remains 68.519 mHa above 6-31G and does not reach polarized double-zeta or triple-zeta energies.
- This is a single-geometry variational comparison. It does not establish transferability, forces, or basis-set-limit accuracy.
