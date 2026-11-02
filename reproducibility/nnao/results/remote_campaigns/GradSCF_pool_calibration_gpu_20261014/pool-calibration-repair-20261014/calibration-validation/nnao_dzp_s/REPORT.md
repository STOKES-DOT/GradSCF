# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical nnao_dzp_s; core policy 6-31g; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 18 / 10 | -1.1272512237 | -1.1270912958 | 0.15993 | max_steps |
| baseline | CH4 | 59 / 34 | -40.1736542640 | -40.1730335525 | 0.62071 | max_steps |
| baseline | NH3 | 50 / 29 | -56.1435896321 | -56.1408306015 | 2.75903 | max_steps |
| baseline | H2O | 41 / 24 | -75.9320254545 | -75.9270345302 | 4.99092 | max_steps |
| baseline | HF | 32 / 19 | -99.8708597302 | -99.8654496441 | 5.41009 | max_steps |
| baseline | CO | 46 / 28 | -112.6498074111 | -112.6441034768 | 5.70393 | max_steps |
| baseline | GMTKN55-v2/SIE4x4/he | 9 / 5 | -2.8356951411 | -2.8356951405 | 0.00000 | rejected |
| baseline | GMTKN55-v2/ALK8/li_h | 19 / 11 | -7.9764618739 | -7.9740479002 | 2.41397 | max_steps |
| baseline | GMTKN55-v2/W4-11/be2 | 20 / 12 | -29.1166076094 | -29.1083970406 | 8.21057 | max_steps |
| baseline | GMTKN55-v2/W4-11/bh | 32 / 19 | -25.1144224246 | -25.1138511730 | 0.57125 | max_steps |
| baseline | GMTKN55-v2/RG18/ne2 | 46 / 28 | -256.5500346350 | -256.5499625048 | 0.07213 | rejected |
| baseline | GMTKN55-v2/MB16-43/NaH | 23 / 15 | -162.3699461438 | -162.3688281057 | 1.11804 | max_steps |
| baseline | GMTKN55-v2/ALKBDE10/mgo | 37 / 24 | -274.2239101935 | -274.2026463823 | 21.26381 | max_steps |
| baseline | GMTKN55-v2/W4-11/alh | 36 / 23 | -242.4363480640 | -242.4346724693 | 1.67559 | max_steps |
| baseline | GMTKN55-v2/G2RC/47 | 50 / 32 | -363.7043859603 | -363.6881258366 | 16.26012 | max_steps |
| baseline | GMTKN55-v2/G21IP/43 | 54 / 36 | -681.4216231675 | -681.4158358807 | 5.78729 | max_steps |
| baseline | GMTKN55-v2/G21IP/48 | 50 / 32 | -435.2831056852 | -435.2762190254 | 6.88666 | max_steps |
| baseline | GMTKN55-v2/AHB21/8B | 36 / 23 | -460.0237177024 | -460.0209025337 | 2.81517 | max_steps |
| baseline | GMTKN55-v2/RG18/ar2 | 54 / 36 | -1053.5240911419 | -1053.5206071620 | 3.48398 | max_steps |
| baseline | GMTKN55-v2/ALKBDE10/kf | 41 / 28 | -698.3934479240 | -698.3787915456 | 14.65638 | max_steps |
| baseline | GMTKN55-v2/ALKBDE10/cao | 41 / 28 | -751.2641288663 | -751.2506943740 | 13.43449 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/ge2h6 | 126 / 84 | -4143.3168938855 | -4142.6284686930 | 688.42519 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/as2me4 | 272 / 170 | -4614.2726162353 | -4613.3697766528 | 902.83958 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/h2se2 | 90 / 64 | -4786.2496243663 | -4785.1922544234 | 1057.36994 | max_steps |
| baseline | GMTKN55-v2/HEAVY28/hbr | 45 / 32 | -2564.7846516863 | -2564.1543118739 | 630.33981 | max_steps |
| baseline | GMTKN55-v2/RG18/kr2 | 72 / 54 | -5485.7639350752 | -5484.2630718095 | 1500.86327 | max_steps |
| calibrated_candidate | H2 | 18 / 10 | -1.1271778663 | -1.1270933227 | 0.08454 | max_steps |
| calibrated_candidate | CH4 | 59 / 34 | -40.1740776568 | -40.1733954930 | 0.68216 | max_steps |
| calibrated_candidate | NH3 | 50 / 29 | -56.1447729028 | -56.1419224308 | 2.85047 | max_steps |
| calibrated_candidate | H2O | 41 / 24 | -75.9345090336 | -75.9297093056 | 4.79973 | max_steps |
| calibrated_candidate | HF | 32 / 19 | -99.8742979242 | -99.8690671414 | 5.23078 | max_steps |
| calibrated_candidate | CO | 46 / 28 | -112.6504563053 | -112.6450793116 | 5.37699 | max_steps |
| calibrated_candidate | GMTKN55-v2/SIE4x4/he | 9 / 5 | -2.8356381907 | -2.8356376068 | 0.00058 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALK8/li_h | 19 / 11 | -7.9766555434 | -7.9739205155 | 2.73503 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/be2 | 20 / 12 | -29.1168096340 | -29.1085510930 | 8.25854 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/bh | 32 / 19 | -25.1145100394 | -25.1138633465 | 0.64669 | max_steps |
| calibrated_candidate | GMTKN55-v2/RG18/ne2 | 46 / 28 | -256.5590012238 | -256.5573667783 | 1.63445 | max_steps |
| calibrated_candidate | GMTKN55-v2/MB16-43/NaH | 23 / 15 | -162.3701353012 | -162.3689247848 | 1.21052 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/mgo | 37 / 24 | -274.2276490818 | -274.2066931468 | 20.95594 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/alh | 36 / 23 | -242.4365167362 | -242.4350075449 | 1.50919 | max_steps |
| calibrated_candidate | GMTKN55-v2/G2RC/47 | 50 / 32 | -363.7050222667 | -363.6896933336 | 15.32893 | max_steps |
| calibrated_candidate | GMTKN55-v2/G21IP/43 | 54 / 36 | -681.4220156132 | -681.4158329714 | 6.18264 | max_steps |
| calibrated_candidate | GMTKN55-v2/G21IP/48 | 50 / 32 | -435.2836008682 | -435.2765980785 | 7.00279 | max_steps |
| calibrated_candidate | GMTKN55-v2/AHB21/8B | 36 / 23 | -460.0241344471 | -460.0212551884 | 2.87926 | max_steps |
| calibrated_candidate | GMTKN55-v2/RG18/ar2 | 54 / 36 | -1053.5251314622 | -1053.5214902558 | 3.64121 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/kf | 41 / 28 | -698.4011838738 | -698.3875494810 | 13.63439 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/cao | 41 / 28 | -751.2696776927 | -751.2561430790 | 13.53461 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/ge2h6 | 126 / 84 | -4143.4135353898 | -4142.7186844388 | 694.85095 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/as2me4 | 272 / 170 | -4614.3783367173 | -4613.4607213931 | 917.61532 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/h2se2 | 90 / 64 | -4786.3633682961 | -4785.3010140572 | 1062.35424 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVY28/hbr | 45 / 32 | -2564.8474747281 | -2564.2185411117 | 628.93362 | max_steps |
| calibrated_candidate | GMTKN55-v2/RG18/kr2 | 72 / 54 | -5485.9021901420 | -5484.4026474821 | 1499.54266 | max_steps |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
