# NNAO primitive-pool / free-contraction pilot

Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.
Spherical nnao_szp_l; core policy 6-31g; fixed def2-universal-jkfit DF approximation.

| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | H2 | 24 / 8 | -1.1292863178 | -1.1292863073 | 0.00001 | converged |
| baseline | CH4 | 88 / 26 | -40.1881587457 | -40.1869622177 | 1.19653 | converged |
| baseline | NH3 | 76 / 22 | -56.1777000582 | -56.1654068266 | 12.29323 | converged |
| baseline | H2O | 64 / 18 | -75.9895992887 | -75.9768580228 | 12.74127 | converged |
| baseline | HF | 52 / 14 | -99.9499755500 | -99.9396588945 | 10.31666 | converged |
| baseline | CO | 80 / 20 | -112.6938652266 | -112.6426083839 | 51.25684 | converged |
| baseline | GMTKN55-v2/SIE4x4/he | 12 / 4 | -2.8434735650 | -2.8434735650 | 0.00000 | converged |
| baseline | GMTKN55-v2/ALK8/li_h | 25 / 9 | -7.9794551811 | -7.9792476080 | 0.20757 | converged |
| baseline | GMTKN55-v2/W4-11/be2 | 26 / 10 | -29.1206660798 | -29.1142651401 | 6.40094 | converged |
| baseline | GMTKN55-v2/W4-11/bh | 52 / 14 | -25.1206988262 | -25.1154337604 | 5.26507 | max_steps |
| baseline | GMTKN55-v2/RG18/ne2 | 80 / 20 | -256.7324074187 | -256.7217543683 | 10.65305 | converged |
| baseline | GMTKN55-v2/MB16-43/NaH | 29 / 13 | -162.3721612947 | -162.3710281367 | 1.13316 | converged |
| baseline | GMTKN55-v2/ALKBDE10/mgo | 57 / 19 | -274.2864657213 | -274.2782981190 | 8.16760 | max_steps |
| baseline | GMTKN55-v2/W4-11/alh | 56 / 18 | -242.4402400042 | -242.4343961167 | 5.84389 | converged |
| baseline | GMTKN55-v2/G2RC/47 | 84 / 24 | -363.7517793068 | -363.7280291259 | 23.75018 | converged |
| baseline | GMTKN55-v2/G21IP/43 | 88 / 28 | -681.4363623341 | -681.4110696638 | 25.29267 | converged |
| baseline | GMTKN55-v2/G21IP/48 | 84 / 24 | -435.3046070002 | -435.2665748804 | 38.03212 | max_steps |
| baseline | GMTKN55-v2/AHB21/8B | 56 / 18 | -460.0350764160 | -460.0263170698 | 8.75935 | converged |
| baseline | GMTKN55-v2/RG18/ar2 | 88 / 28 | -1053.5469311680 | -1053.5287581099 | 18.17306 | max_steps |
| baseline | GMTKN55-v2/ALKBDE10/kf | 61 / 23 | -698.5173065045 | -698.5068038922 | 10.50261 | rejected |
| baseline | GMTKN55-v2/ALKBDE10/cao | 61 / 23 | -751.3541677190 | -751.3366115010 | 17.55622 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/ge2h6 | 178 / 70 | -4149.6362331505 | -4149.4352284599 | 201.00469 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/as2me4 | 410 / 134 | -4621.7819995051 | -4621.4664948847 | 315.50462 | max_steps |
| baseline | GMTKN55-v2/HEAVYSB11/h2se2 | 130 / 54 | -4794.8518817540 | -4794.6437390221 | 208.14273 | max_steps |
| baseline | GMTKN55-v2/HEAVY28/hbr | 65 / 27 | -2569.6379772565 | -2569.5828016502 | 55.17561 | rejected |
| baseline | GMTKN55-v2/RG18/kr2 | 106 / 46 | -5496.6242066699 | -5496.5607125241 | 63.49415 | max_steps |
| calibrated_candidate | H2 | 24 / 8 | -1.1300818309 | -1.1300817222 | 0.00011 | max_steps |
| calibrated_candidate | CH4 | 88 / 26 | -40.1944417704 | -40.1932501124 | 1.19166 | max_steps |
| calibrated_candidate | NH3 | 76 / 22 | -56.1902604617 | -56.1799956094 | 10.26485 | max_steps |
| calibrated_candidate | H2O | 64 / 18 | -76.0122970938 | -76.0024494679 | 9.84763 | max_steps |
| calibrated_candidate | HF | 52 / 14 | -99.9892672597 | -99.9816951498 | 7.57211 | converged |
| calibrated_candidate | CO | 80 / 20 | -112.7221650683 | -112.6772416433 | 44.92342 | converged |
| calibrated_candidate | GMTKN55-v2/SIE4x4/he | 12 / 4 | -2.8457028878 | -2.8457028012 | 0.00009 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALK8/li_h | 25 / 9 | -7.9795769298 | -7.9793000015 | 0.27693 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/be2 | 26 / 10 | -29.1229498940 | -29.1161310402 | 6.81885 | converged |
| calibrated_candidate | GMTKN55-v2/W4-11/bh | 52 / 14 | -25.1231636473 | -25.1175897580 | 5.57389 | max_steps |
| calibrated_candidate | GMTKN55-v2/RG18/ne2 | 80 / 20 | -256.8605794300 | -256.8503577996 | 10.22163 | converged |
| calibrated_candidate | GMTKN55-v2/MB16-43/NaH | 29 / 13 | -162.3724806880 | -162.3711056974 | 1.37499 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/mgo | 57 / 19 | -274.3100757676 | -274.3020484661 | 8.02730 | max_steps |
| calibrated_candidate | GMTKN55-v2/W4-11/alh | 56 / 18 | -242.4410350626 | -242.4356583394 | 5.37672 | max_steps |
| calibrated_candidate | GMTKN55-v2/G2RC/47 | 84 / 24 | -363.7749357744 | -363.7552172494 | 19.71853 | max_steps |
| calibrated_candidate | GMTKN55-v2/G21IP/43 | 88 / 28 | -681.4381399405 | -681.4128293664 | 25.31057 | max_steps |
| calibrated_candidate | GMTKN55-v2/G21IP/48 | 84 / 24 | -435.3110298858 | -435.2744634643 | 36.56642 | max_steps |
| calibrated_candidate | GMTKN55-v2/AHB21/8B | 56 / 18 | -460.0369857181 | -460.0281721911 | 8.81353 | converged |
| calibrated_candidate | GMTKN55-v2/RG18/ar2 | 88 / 28 | -1053.5499532669 | -1053.5306797230 | 19.27354 | converged |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/kf | 61 / 23 | -698.5560079600 | -698.5456879179 | 10.32004 | max_steps |
| calibrated_candidate | GMTKN55-v2/ALKBDE10/cao | 61 / 23 | -751.3794011677 | -751.1842080755 | 195.19309 | rejected |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/ge2h6 | 178 / 70 | -4151.4690450937 | -4151.3320797201 | 136.96537 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/as2me4 | 410 / 134 | -4623.9108473359 | -4623.6664907241 | 244.35661 | converged |
| calibrated_candidate | GMTKN55-v2/HEAVYSB11/h2se2 | 130 / 54 | -4797.3627688061 | -4797.2151296415 | 147.63916 | max_steps |
| calibrated_candidate | GMTKN55-v2/HEAVY28/hbr | 65 / 27 | -2571.0690575999 | -2571.0330934952 | 35.96410 | rejected |
| calibrated_candidate | GMTKN55-v2/RG18/kr2 | 106 / 46 | -5499.8052533026 | -5499.7724783855 | 32.77492 | max_steps |

## Interpretation

These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.
No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.
No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.
The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.
RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.
