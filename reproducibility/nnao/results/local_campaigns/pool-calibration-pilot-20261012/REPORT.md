# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical szp663_direct/core6; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 24 / 8 | -1.1292863178 | -1.1292228449 | 0.06347 | max_steps |
| baseline | CH4 | 88 / 26 | -40.1881587439 | -40.1827243860 | 5.43436 | max_steps |
| baseline | NH3 | 76 / 22 | -56.1777000347 | -56.1546268896 | 23.07315 | max_steps |
| baseline | H2O | 64 / 18 | -75.9895992904 | -75.9599612078 | 29.63808 | max_steps |
| baseline | HF | 52 / 14 | -99.9499755500 | -99.9229378362 | 27.03771 | max_steps |
| baseline | CO | 80 / 20 | -112.6938652300 | -112.6160519524 | 77.81328 | max_steps |
| calibrated_candidate | H2 | 24 / 8 | -1.1290293973 | -1.1289342943 | 0.09510 | max_steps |
| calibrated_candidate | CH4 | 88 / 26 | -40.1906683439 | -40.1851237735 | 5.54457 | max_steps |
| calibrated_candidate | NH3 | 76 / 22 | -56.1825923206 | -56.1599245274 | 22.66779 | max_steps |
| calibrated_candidate | H2O | 64 / 18 | -75.9976889057 | -75.9720387203 | 25.65019 | max_steps |
| calibrated_candidate | HF | 52 / 14 | -99.9637362847 | -99.9405552153 | 23.18107 | max_steps |
| calibrated_candidate | CO | 80 / 20 | -112.7037459133 | -112.6311013335 | 72.64458 | max_steps |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
