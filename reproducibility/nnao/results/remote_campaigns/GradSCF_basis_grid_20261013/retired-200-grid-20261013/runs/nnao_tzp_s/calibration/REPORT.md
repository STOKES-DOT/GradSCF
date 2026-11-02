# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical nnao_tzp_s/core6; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 18 / 12 | -1.1272512237 | -1.1272512110 | 0.00001 | converged |
| baseline | CH4 | 59 / 42 | -40.1736542738 | -40.1735642206 | 0.09005 | converged |
| baseline | NH3 | 50 / 36 | -56.1435894024 | -56.1427217927 | 0.86761 | rejected |
| baseline | H2O | 41 / 30 | -75.9320254779 | -75.9303594478 | 1.66603 | converged |
| baseline | HF | 32 / 24 | -99.8708597302 | -99.8680300154 | 2.82971 | converged |
| baseline | CO | 46 / 36 | -112.6498074531 | -112.6474688322 | 2.33862 | converged |
| calibrated_candidate | H2 | 18 / 12 | -1.1274078617 | -1.1274078432 | 0.00002 | converged |
| calibrated_candidate | CH4 | 59 / 42 | -40.1766369157 | -40.1764249542 | 0.21196 | converged |
| calibrated_candidate | NH3 | 50 / 36 | -56.1482104720 | -56.1470594851 | 1.15099 | rejected |
| calibrated_candidate | H2O | 41 / 30 | -75.9396522869 | -75.9373029055 | 2.34938 | converged |
| calibrated_candidate | HF | 32 / 24 | -99.8824645527 | -99.8793180005 | 3.14655 | converged |
| calibrated_candidate | CO | 46 / 36 | -112.6603404540 | -112.6580098462 | 2.33061 | converged |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
