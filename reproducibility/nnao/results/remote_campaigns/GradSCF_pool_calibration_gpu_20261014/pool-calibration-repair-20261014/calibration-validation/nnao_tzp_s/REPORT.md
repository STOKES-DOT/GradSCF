# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical nnao_tzp_s; core policy 6-31g; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 18 / 12 | -1.1272512237 | -1.1271874033 | 0.06382 | max_steps |
| baseline | CH4 | 59 / 42 | -40.1736542640 | -40.1730745367 | 0.57973 | max_steps |
| baseline | NH3 | 50 / 36 | -56.1435896321 | -56.1412733079 | 2.31632 | max_steps |
| baseline | H2O | 41 / 30 | -75.9320254545 | -75.9277470918 | 4.27836 | max_steps |
| baseline | HF | 32 / 24 | -99.8708597302 | -99.8660882593 | 4.77147 | max_steps |
| baseline | CO | 46 / 36 | -112.6498074111 | -112.6446857794 | 5.12163 | max_steps |
| baseline | GMTKN55-v2/SIE4x4/he | 9 / 6 | -2.8356951411 | -2.8356951411 | -0.00000 | converged |
| baseline | GMTKN55-v2/ALK8/li_h | 19 / 13 | -7.9764618739 | -7.9744035536 | 2.05832 | max_steps |
| baseline | GMTKN55-v2/W4-11/be2 | 20 / 14 | -29.1166076094 | -29.1084452370 | 8.16237 | max_steps |
| baseline | GMTKN55-v2/W4-11/bh | 32 / 24 | -25.1144224246 | -25.1140274590 | 0.39497 | max_steps |
| baseline | GMTKN55-v2/RG18/ne2 | 46 / 36 | -256.5500346350 | -256.5500335446 | 0.00109 | converged |
| baseline | GMTKN55-v2/MB16-43/NaH | 23 / 17 | -162.3699461438 | -162.3696010045 | 0.34514 | max_steps |
| baseline | GMTKN55-v2/ALKBDE10/mgo | 37 / 29 | -274.2239101935 | -274.2042499698 | 19.66022 | max_steps |
| baseline | GMTKN55-v2/W4-11/alh | 36 / 28 | -242.4363480640 | -242.4356979951 | 0.65007 | max_steps |
| baseline | GMTKN55-v2/G2RC/47 | 50 / 40 | -363.7043859603 | -363.6896070853 | 14.77887 | max_steps |
| baseline | GMTKN55-v2/G21IP/43 | 54 / 44 | -681.4216231675 | -681.4184407606 | 3.18241 | max_steps |
| baseline | GMTKN55-v2/G21IP/48 | 50 / 40 | -435.2831056852 | -435.2780309771 | 5.07471 | max_steps |
| baseline | GMTKN55-v2/AHB21/8B | 36 / 28 | -460.0237177024 | -460.0229252554 | 0.79245 | max_steps |
| baseline | GMTKN55-v2/RG18/ar2 | 54 / 44 | -1053.5240911419 | -1053.5240337477 | 0.05739 | max_steps |
| baseline | GMTKN55-v2/ALKBDE10/kf | 41 / 33 | -698.3934479240 | -698.3833694730 | 10.07845 | max_steps |
| baseline | GMTKN55-v2/ALKBDE10/cao | 41 / 33 | -751.2641288663 | -751.2523959519 | 11.73291 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/ge2h6 | 126 / 98 | -4143.3168938855 | -4142.7599287276 | 556.96516 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/as2me4 | 272 / 206 | -4614.2726162353 | -4613.5288827177 | 743.73352 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/h2se2 | 90 / 74 | -4786.2496243663 | -4785.3761422957 | 873.48207 | max_steps |
| baseline | GMTKN55-v2/HEAVY28/hbr | 45 / 37 | -2564.7846516863 | -2564.2595696485 | 525.08204 | max_steps |
| baseline | GMTKN55-v2/RG18/kr2 | 72 / 62 | -5485.7639350752 | -5484.4881530574 | 1275.78202 | max_steps |
| calibrated_candidate | H2 | 18 / 12 | -1.1271778663 | -1.1271154297 | 0.06244 | max_steps |
| calibrated_candidate | CH4 | 59 / 42 | -40.1740776568 | -40.1734553721 | 0.62228 | max_steps |
| calibrated_candidate | NH3 | 50 / 36 | -56.1447729028 | -56.1424651371 | 2.30777 | max_steps |
| calibrated_candidate | H2O | 41 / 30 | -75.9345090336 | -75.9302395596 | 4.26947 | max_steps |
| calibrated_candidate | HF | 32 / 24 | -99.8742979242 | -99.8696049989 | 4.69293 | max_steps |
| calibrated_candidate | CO | 46 / 36 | -112.6504563053 | -112.6457209562 | 4.73535 | max_steps |
| calibrated_candidate | GMTKN55-v2/SIE4x4/he | 9 / 6 | -2.8356381907 | -2.8356381907 | 0.00000 | converged |
| calibrated_candidate | GMTKN55-v2/ALK8/li_h | 19 / 13 | -7.9766555434 | -7.9744057816 | 2.24976 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/be2 | 20 / 14 | -29.1168096340 | -29.1087220293 | 8.08760 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/bh | 32 / 24 | -25.1145100394 | -25.1140462398 | 0.46380 | max_steps |
| calibrated_candidate | GMTKN55-v2/RG18/ne2 | 46 / 36 | -256.5590012238 | -256.5589999699 | 0.00125 | converged |
| calibrated_candidate | GMTKN55-v2/MB16-43/NaH | 23 / 17 | -162.3701353012 | -162.3697572233 | 0.37808 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/mgo | 37 / 29 | -274.2276490818 | -274.2085289641 | 19.12012 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/alh | 36 / 28 | -242.4365167362 | -242.4359235617 | 0.59317 | max_steps |
| calibrated_candidate | GMTKN55-v2/G2RC/47 | 50 / 40 | -363.7050222667 | -363.6911654872 | 13.85678 | max_steps |
| calibrated_candidate | GMTKN55-v2/G21IP/43 | 54 / 44 | -681.4220156132 | -681.4184182416 | 3.59737 | max_steps |
| calibrated_candidate | GMTKN55-v2/G21IP/48 | 50 / 40 | -435.2836008682 | -435.2783982515 | 5.20262 | max_steps |
| calibrated_candidate | GMTKN55-v2/AHB21/8B | 36 / 28 | -460.0241344471 | -460.0232661304 | 0.86832 | max_steps |
| calibrated_candidate | GMTKN55-v2/RG18/ar2 | 54 / 44 | -1053.5251314622 | -1053.5250724474 | 0.05901 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/kf | 41 / 33 | -698.4011838738 | -698.3921472870 | 9.03659 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/cao | 41 / 33 | -751.2696776927 | -751.2578337466 | 11.84395 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/ge2h6 | 126 / 98 | -4143.4135353898 | -4142.8490963735 | 564.43902 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/as2me4 | 272 / 206 | -4614.3783367173 | -4613.6190013170 | 759.33540 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/h2se2 | 90 / 74 | -4786.3633682961 | -4785.4825865174 | 880.78178 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVY28/hbr | 45 / 37 | -2564.8474747281 | -2564.3227865548 | 524.68817 | max_steps |
| calibrated_candidate | GMTKN55-v2/RG18/kr2 | 72 / 62 | -5485.9021901420 | -5484.6250249832 | 1277.16516 | max_steps |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
