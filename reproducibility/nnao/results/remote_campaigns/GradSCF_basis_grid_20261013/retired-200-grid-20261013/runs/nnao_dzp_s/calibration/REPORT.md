# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical nnao_dzp_s/core6; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 18 / 10 | -1.1272512237 | -1.1272510500 | 0.00017 | rejected |
| baseline | CH4 | 59 / 34 | -40.1736542738 | -40.1735636712 | 0.09060 | converged |
| baseline | NH3 | 50 / 29 | -56.1435894024 | -56.1426402400 | 0.94916 | rejected |
| baseline | H2O | 41 / 24 | -75.9320254779 | -75.9302387353 | 1.78674 | rejected |
| baseline | HF | 32 / 19 | -99.8708597302 | -99.8679222991 | 2.93743 | rejected |
| baseline | CO | 46 / 28 | -112.6498074531 | -112.6471360051 | 2.67145 | max_steps |
| calibrated_candidate | H2 | 18 / 10 | -1.1274078617 | -1.1274077945 | 0.00007 | rejected |
| calibrated_candidate | CH4 | 59 / 34 | -40.1766369157 | -40.1764247984 | 0.21212 | converged |
| calibrated_candidate | NH3 | 50 / 29 | -56.1482104720 | -56.1469812140 | 1.22926 | rejected |
| calibrated_candidate | H2O | 41 / 24 | -75.9396522869 | -75.9371794687 | 2.47282 | rejected |
| calibrated_candidate | HF | 32 / 19 | -99.8824645527 | -99.8792061640 | 3.25839 | converged |
| calibrated_candidate | CO | 46 / 28 | -112.6603404540 | -112.6578227092 | 2.51774 | max_steps |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
