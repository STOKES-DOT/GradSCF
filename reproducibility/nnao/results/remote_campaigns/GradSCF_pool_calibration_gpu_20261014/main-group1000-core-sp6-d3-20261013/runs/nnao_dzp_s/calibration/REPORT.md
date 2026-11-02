# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical nnao_dzp_s; core policy 6-31g; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 18 / 10 | -1.1272512237 | -1.1272510500 | 0.00017 | rejected |
| baseline | CH4 | 59 / 34 | -40.1736542640 | -40.1735636614 | 0.09060 | converged |
| baseline | NH3 | 50 / 29 | -56.1435896321 | -56.1426404204 | 0.94921 | rejected |
| baseline | H2O | 41 / 24 | -75.9320254545 | -75.9302387196 | 1.78673 | rejected |
| baseline | HF | 32 / 19 | -99.8708597302 | -99.8679222991 | 2.93743 | rejected |
| baseline | CO | 46 / 28 | -112.6498074111 | -112.6471359669 | 2.67144 | max_steps |
| baseline | GMTKN55-v2/SIE4x4/he | 9 / 5 | -2.8356951411 | -2.8356951405 | 0.00000 | rejected |
| baseline | GMTKN55-v2/ALK8/li_h | 19 / 11 | -7.9764618739 | -7.9763624669 | 0.09941 | rejected |
| baseline | GMTKN55-v2/W4-11/be2 | 20 / 12 | -29.1166076094 | -29.1158774763 | 0.73013 | rejected |
| baseline | GMTKN55-v2/W4-11/bh | 32 / 19 | -25.1144224246 | -25.1143646805 | 0.05774 | rejected |
| baseline | GMTKN55-v2/RG18/ne2 | 46 / 28 | -256.5500346350 | -256.5499625048 | 0.07213 | rejected |
| baseline | GMTKN55-v2/MB16-43/NaH | 23 / 15 | -162.3699461438 | -162.3697928310 | 0.15331 | rejected |
| baseline | GMTKN55-v2/ALKBDE10/mgo | 37 / 24 | -274.2239101935 | -274.2171523643 | 6.75783 | converged |
| baseline | GMTKN55-v2/W4-11/alh | 36 / 23 | -242.4363480640 | -242.4352824758 | 1.06559 | rejected |
| baseline | GMTKN55-v2/G2RC/47 | 50 / 32 | -363.7043859603 | -363.6990962311 | 5.28973 | max_steps |
| baseline | GMTKN55-v2/G21IP/43 | 54 / 36 | -681.4216231675 | -681.4185563957 | 3.06677 | converged |
| baseline | GMTKN55-v2/G21IP/48 | 50 / 32 | -435.2831056852 | -435.2799758635 | 3.12982 | converged |
| baseline | GMTKN55-v2/AHB21/8B | 36 / 23 | -460.0237177024 | -460.0224605872 | 1.25712 | rejected |
| baseline | GMTKN55-v2/RG18/ar2 | 54 / 36 | -1053.5240911419 | -1053.5222804443 | 1.81070 | rejected |
| baseline | GMTKN55-v2/ALKBDE10/kf | 41 / 28 | -698.3934479240 | -698.3848887336 | 8.55919 | rejected |
| baseline | GMTKN55-v2/ALKBDE10/cao | 41 / 28 | -751.2641288663 | -751.2613091776 | 2.81969 | rejected |
| baseline | GMTKN55-v2/HEAVYSB11/ge2h6 | 126 / 84 | -4143.3168938855 | -4143.2245907763 | 92.30311 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/as2me4 | 272 / 170 | -4614.2726162353 | -4614.1374016563 | 135.21458 | converged |
| baseline | GMTKN55-v2/HEAVYSB11/h2se2 | 90 / 64 | -4786.2496243663 | -4786.1675729782 | 82.05139 | converged |
| baseline | GMTKN55-v2/HEAVY28/hbr | 45 / 32 | -2564.7846516863 | -2564.7706886804 | 13.96301 | rejected |
| baseline | GMTKN55-v2/RG18/kr2 | 72 / 54 | -5485.7639350752 | -5485.7591787893 | 4.75629 | converged |
| calibrated_candidate | H2 | 18 / 10 | -1.1254459999 | -1.1254459997 | 0.00000 | converged |
| calibrated_candidate | CH4 | 59 / 34 | -40.1764384307 | -40.1763677643 | 0.07067 | converged |
| calibrated_candidate | NH3 | 50 / 29 | -56.1461753399 | -56.1446989997 | 1.47634 | converged |
| calibrated_candidate | H2O | 41 / 24 | -75.9370282466 | -75.9355289978 | 1.49925 | converged |
| calibrated_candidate | HF | 32 / 19 | -99.8776371365 | -99.8761965365 | 1.44060 | converged |
| calibrated_candidate | CO | 46 / 28 | -112.6572357996 | -112.6528722806 | 4.36352 | rejected |
| calibrated_candidate | GMTKN55-v2/SIE4x4/he | 9 / 5 | -2.8356910249 | -2.8356910247 | 0.00000 | rejected |
| calibrated_candidate | GMTKN55-v2/ALK8/li_h | 19 / 11 | -7.9781206916 | -7.9780237291 | 0.09696 | converged |
| calibrated_candidate | GMTKN55-v2/W4-11/be2 | 20 / 12 | -29.1182592614 | -29.1173811351 | 0.87813 | rejected |
| calibrated_candidate | GMTKN55-v2/W4-11/bh | 32 / 19 | -25.1144971971 | -25.1143929230 | 0.10427 | rejected |
| calibrated_candidate | GMTKN55-v2/RG18/ne2 | 46 / 28 | -256.5742313977 | -256.5742306210 | 0.00078 | max_steps |
| calibrated_candidate | GMTKN55-v2/MB16-43/NaH | 23 / 15 | -162.3708133832 | -162.3706010986 | 0.21228 | converged |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/mgo | 37 / 24 | -274.2343067079 | -274.2290415376 | 5.26517 | rejected |
| calibrated_candidate | GMTKN55-v2/W4-11/alh | 36 / 23 | -242.4374269304 | -242.4364739304 | 0.95300 | converged |
| calibrated_candidate | GMTKN55-v2/G2RC/47 | 50 / 32 | -363.7087724978 | -363.7029021810 | 5.87032 | rejected |
| calibrated_candidate | GMTKN55-v2/G21IP/43 | 54 / 36 | -681.4248305977 | -681.4216422380 | 3.18836 | converged |
| calibrated_candidate | GMTKN55-v2/G21IP/48 | 50 / 32 | -435.2880263762 | -435.2856560708 | 2.37031 | converged |
| calibrated_candidate | GMTKN55-v2/AHB21/8B | 36 / 23 | -460.0282633590 | -460.0261991613 | 2.06420 | rejected |
| calibrated_candidate | GMTKN55-v2/RG18/ar2 | 54 / 36 | -1053.5335936814 | -1053.5311350159 | 2.45867 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/kf | 41 / 28 | -698.4189316052 | -698.4110967665 | 7.83484 | rejected |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/cao | 41 / 28 | -751.2812725802 | -751.2679339913 | 13.33859 | rejected |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/ge2h6 | 126 / 84 | -4144.8300810563 | -4144.6564758442 | 173.60521 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/as2me4 | 272 / 170 | -4615.9567837718 | -4615.7529292015 | 203.85457 | converged |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/h2se2 | 90 / 64 | -4788.1113391765 | -4787.9685084115 | 142.83076 | converged |
| calibrated_candidate | GMTKN55-v2/HEAVY28/hbr | 45 / 32 | -2565.8100666250 | -2565.7947882398 | 15.27839 | converged |
| calibrated_candidate | GMTKN55-v2/RG18/kr2 | 72 / 54 | -5488.0250963786 | -5488.0228943966 | 2.20198 | converged |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
