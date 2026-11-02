# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical szp663_direct/core6; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 24 / 8 | -1.1292863178 | -1.1292863073 | 0.00001 | converged |
| baseline | CH4 | 88 / 26 | -40.1881587439 | -40.1869622193 | 1.19652 | converged |
| baseline | NH3 | 76 / 22 | -56.1777000347 | -56.1654067517 | 12.29328 | converged |
| baseline | H2O | 64 / 18 | -75.9895992904 | -75.9768580285 | 12.74126 | converged |
| baseline | HF | 52 / 14 | -99.9499755500 | -99.9396588945 | 10.31666 | converged |
| baseline | CO | 80 / 20 | -112.6938652300 | -112.6426084053 | 51.25682 | converged |
| calibrated_candidate | H2 | 24 / 8 | -1.1298975464 | -1.1298975458 | 0.00000 | converged |
| calibrated_candidate | CH4 | 88 / 26 | -40.1948264552 | -40.1935445322 | 1.28192 | converged |
| calibrated_candidate | NH3 | 76 / 22 | -56.1907720193 | -56.1803321096 | 10.43991 | converged |
| calibrated_candidate | H2O | 64 / 18 | -76.0127024671 | -76.0023915408 | 10.31093 | converged |
| calibrated_candidate | HF | 52 / 14 | -99.9895882448 | -99.9817594638 | 7.82878 | converged |
| calibrated_candidate | CO | 80 / 20 | -112.7222259693 | -112.6770845041 | 45.14147 | converged |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
