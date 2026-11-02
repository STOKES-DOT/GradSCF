# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical nnao_dzp_m; core policy 6-31g; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 20 / 10 | -1.1279119507 | -1.1278729056 | 0.03905 | rejected |
| baseline | CH4 | 67 / 34 | -40.1756233297 | -40.1755319270 | 0.09140 | max_steps |
| baseline | NH3 | 57 / 29 | -56.1582466166 | -56.1572129748 | 1.03364 | max_steps |
| baseline | H2O | 47 / 24 | -75.9557696785 | -75.9538689527 | 1.90073 | max_steps |
| baseline | HF | 37 / 19 | -99.8984773775 | -99.8962835740 | 2.19380 | max_steps |
| baseline | CO | 54 / 28 | -112.6592605962 | -112.6548819031 | 4.37869 | rejected |
| baseline | GMTKN55-v2/SIE4x4/he | 10 / 5 | -2.8378912634 | -2.8378734968 | 0.01777 | rejected |
| baseline | GMTKN55-v2/ALK8/li_h | 21 / 11 | -7.9768229778 | -7.9767210600 | 0.10192 | rejected |
| baseline | GMTKN55-v2/W4-11/be2 | 22 / 12 | -29.1174420048 | -29.1168124941 | 0.62951 | rejected |
| baseline | GMTKN55-v2/W4-11/bh | 37 / 19 | -25.1166880176 | -25.1166134282 | 0.07459 | converged |
| baseline | GMTKN55-v2/RG18/ne2 | 54 / 28 | -256.5985429331 | -256.5981888112 | 0.35412 | max_steps |
| baseline | GMTKN55-v2/MB16-43/NaH | 25 / 15 | -162.3702254648 | -162.3689416741 | 1.28379 | rejected |
| baseline | GMTKN55-v2/ALKBDE10/mgo | 42 / 24 | -274.2554608760 | -274.2519294556 | 3.53142 | max_steps |
| baseline | GMTKN55-v2/W4-11/alh | 41 / 23 | -242.4374431036 | -242.4363394049 | 1.10370 | converged |
| baseline | GMTKN55-v2/G2RC/47 | 58 / 32 | -363.7154038434 | -363.7105380521 | 4.86579 | max_steps |
| baseline | GMTKN55-v2/G21IP/43 | 62 / 36 | -681.4266387303 | -681.4228027943 | 3.83594 | converged |
| baseline | GMTKN55-v2/G21IP/48 | 58 / 32 | -435.2890818063 | -435.2849434818 | 4.13832 | max_steps |
| baseline | GMTKN55-v2/AHB21/8B | 41 / 23 | -460.0311483147 | -460.0292362473 | 1.91207 | converged |
| baseline | GMTKN55-v2/RG18/ar2 | 62 / 36 | -1053.5410635305 | -1053.5375049550 | 3.55858 | max_steps |
| baseline | GMTKN55-v2/ALKBDE10/kf | 46 / 28 | -698.4599756599 | -698.4542959510 | 5.67971 | rejected |
| baseline | GMTKN55-v2/ALKBDE10/cao | 46 / 28 | -751.3152924181 | -751.3096684025 | 5.62402 | rejected |
| baseline | GMTKN55-v2/HEAVYSB11/ge2h6 | 140 / 84 | -4143.3462152763 | -4143.2315413099 | 114.67397 | rejected |
| baseline | GMTKN55-v2/HEAVYSB11/as2me4 | 308 / 170 | -4614.3984712826 | -4614.1545221307 | 243.94915 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/h2se2 | 100 / 64 | -4786.2943555611 | -4786.1748571583 | 119.49840 | max_steps |
| baseline | GMTKN55-v2/HEAVY28/hbr | 50 / 32 | -2564.7994269842 | -2564.7792261386 | 20.20085 | max_steps |
| baseline | GMTKN55-v2/RG18/kr2 | 80 / 54 | -5485.7907967460 | -5485.7804141820 | 10.38256 | max_steps |
| calibrated_candidate | H2 | 20 / 10 | -1.1244564629 | -1.1244564626 | 0.00000 | converged |
| calibrated_candidate | CH4 | 67 / 34 | -40.1777013623 | -40.1776084828 | 0.09288 | max_steps |
| calibrated_candidate | NH3 | 57 / 29 | -56.1639667376 | -56.1625250351 | 1.44170 | max_steps |
| calibrated_candidate | H2O | 47 / 24 | -75.9663074717 | -75.9645322423 | 1.77523 | max_steps |
| calibrated_candidate | HF | 37 / 19 | -99.9184582175 | -99.9132245132 | 5.23370 | rejected |
| calibrated_candidate | CO | 54 / 28 | -112.6833270526 | -112.6807757446 | 2.55131 | max_steps |
| calibrated_candidate | GMTKN55-v2/SIE4x4/he | 10 / 5 | -2.8384185255 | -2.8383805037 | 0.03802 | rejected |
| calibrated_candidate | GMTKN55-v2/ALK8/li_h | 21 / 11 | -7.9785242034 | -7.9784403157 | 0.08389 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/be2 | 22 / 12 | -29.1201810140 | -29.1198121677 | 0.36885 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/bh | 37 / 19 | -25.1172192439 | -25.1171366549 | 0.08259 | converged |
| calibrated_candidate | GMTKN55-v2/RG18/ne2 | 54 / 28 | -256.6692772881 | -256.6690541814 | 0.22311 | max_steps |
| calibrated_candidate | GMTKN55-v2/MB16-43/NaH | 25 / 15 | -162.3706948436 | -162.3704944997 | 0.20034 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/mgo | 42 / 24 | -274.2681115175 | -274.2648380044 | 3.27351 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/alh | 41 / 23 | -242.4382778263 | -242.4372663022 | 1.01152 | max_steps |
| calibrated_candidate | GMTKN55-v2/G2RC/47 | 58 / 32 | -363.7362496996 | -363.7211359286 | 15.11377 | rejected |
| calibrated_candidate | GMTKN55-v2/G21IP/43 | 62 / 36 | -681.4316822185 | -681.4275130022 | 4.16922 | converged |
| calibrated_candidate | GMTKN55-v2/G21IP/48 | 58 / 32 | -435.2976882346 | -435.2923239087 | 5.36433 | converged |
| calibrated_candidate | GMTKN55-v2/AHB21/8B | 41 / 23 | -460.0327292393 | -460.0306225539 | 2.10669 | max_steps |
| calibrated_candidate | GMTKN55-v2/RG18/ar2 | 62 / 36 | -1053.5456523593 | -1053.5428631156 | 2.78924 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/kf | 46 / 28 | -698.4717089398 | -698.4684077132 | 3.30123 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/cao | 46 / 28 | -751.3252994113 | -751.3189795767 | 6.31983 | rejected |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/ge2h6 | 140 / 84 | -4144.8563356747 | -4144.6605888698 | 195.74680 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/as2me4 | 308 / 170 | -4616.0795169554 | -4615.8522144856 | 227.30247 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/h2se2 | 100 / 64 | -4788.1530558740 | -4787.9903920703 | 162.66380 | rejected |
| calibrated_candidate | GMTKN55-v2/HEAVY28/hbr | 50 / 32 | -2565.8297145881 | -2565.7480697382 | 81.64485 | rejected |
| calibrated_candidate | GMTKN55-v2/RG18/kr2 | 80 / 54 | -5488.0409572528 | -5488.0334392838 | 7.51797 | rejected |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
