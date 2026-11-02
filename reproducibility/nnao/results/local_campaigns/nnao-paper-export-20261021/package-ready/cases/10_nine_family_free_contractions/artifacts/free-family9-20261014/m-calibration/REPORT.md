# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical nnao_dzp_m; core policy 6; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 20 / 10 | -1.1279119507 | -1.1278729056 | 0.03905 | rejected |
| baseline | CH4 | 67 / 34 | -40.1756233284 | -40.1755319256 | 0.09140 | max_steps |
| baseline | NH3 | 57 / 29 | -56.1582465813 | -56.1572128766 | 1.03370 | max_steps |
| baseline | H2O | 47 / 24 | -75.9557696816 | -75.9538689595 | 1.90072 | max_steps |
| baseline | HF | 37 / 19 | -99.8984773775 | -99.8962835740 | 2.19380 | max_steps |
| baseline | CO | 54 / 28 | -112.6592606014 | -112.6548819246 | 4.37868 | rejected |
| calibrated_candidate | H2 | 20 / 10 | -1.1279064540 | -1.1279064539 | 0.00000 | converged |
| calibrated_candidate | CH4 | 67 / 34 | -40.1816029371 | -40.1813847420 | 0.21820 | converged |
| calibrated_candidate | NH3 | 57 / 29 | -56.1677807690 | -56.1666630327 | 1.11774 | max_steps |
| calibrated_candidate | H2O | 47 / 24 | -75.9707334511 | -75.9689115061 | 1.82194 | max_steps |
| calibrated_candidate | HF | 37 / 19 | -99.9223294925 | -99.9188938313 | 3.43566 | max_steps |
| calibrated_candidate | CO | 54 / 28 | -112.6850812361 | -112.6823936648 | 2.68757 | rejected |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
