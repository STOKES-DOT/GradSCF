# Ground-state CC references and implemented equations

Theory, source-code adaptation, and test oracles are recorded separately.
BibTeX entries are in [references.bib](references.bib); the source version,
modified routines, hashes, and license are in [NOTICE.md](NOTICE.md).

| Method or component | Reference / definition | Evidence in this implementation |
| --- | --- | --- |
| Restricted CCSD residual and intermediates | Hirata et al. (2004), equations identified in the upstream RCCSD implementation; general theory in Bartlett–Musiał (2007) | Random-amplitude comparison to PySCF 2.9.0 plus independently assembled determinant-space exp(-T) H exp(T) |
| CCS and CCD | Restrict T to T1 or T2 and project only the corresponding residual equations | Independent determinant-space similarity-transform check |
| LCCSD and LCCD | Retain R(0) + DR(0) T and the corresponding constant/linear CC energy; project the selected cluster space | Independent H + [H,T] determinant-space check; these are not CI or quadratic CI |
| CC2 | Christiansen, Koch and Jørgensen (1995); restricted working contractions from PySCF RCCSD's cc2 branch | Random amplitudes and converged energy compared with that explicit CC2 reference |
| Conventional CCSD(T) | Raghavachari et al. (1989); real restricted working formula adapted from PySCF's slow (T) implementation | H4 and H2O corrections compared with PySCF; full energy derivatives compared with finite differences |
| CCSD+T(CCSD) | Urban et al. (1985); pure connected WT2 term distinguished in OpenMolcas CCT3 | Independent determinant-space triple moments and a connected-only PySCF oracle |
| Lambda/implicit response | Stationary Lagrangian E + lambda^T R; general CC response theory in Bartlett–Musiał (2007) | Nonredundant coordinate adjoint, spin-adapted dual conversion, and PySCF CCSD l1/l2 comparison |
| UCCSD / UCCD | Spin-orbital CCSD equations; PySCF `gintermediates` cites Gauss–Stanton (1995), Table III | PySCF UCCSD energies and spin-block amplitudes, random-amplitude GCCSD residuals, and implicit-response finite differences; see [open-shell conventions](OPEN_SHELL.md) |

## Bibliography

1. S. Hirata, R. Podeszwa, M. Tobita, and R. J. Bartlett,
   “Coupled-cluster singles and doubles for extended systems,”
   *J. Chem. Phys.* **120**, 2581–2592 (2004).
   [DOI: 10.1063/1.1637577](https://doi.org/10.1063/1.1637577).
   The [author's publication record](https://hirata-lab.chemistry.illinois.edu/publications.html)
   provides the bibliographic entry. The executable contractions were inspected
   in the attributed upstream source, not reconstructed from a purported
   full-text audit of the original article.
2. O. Christiansen, H. Koch, and P. Jørgensen,
   “The second-order approximate coupled cluster singles and doubles model CC2,”
   *Chem. Phys. Lett.* **243**, 409–418 (1995).
   [DOI: 10.1016/0009-2614(95)00841-Q](https://doi.org/10.1016/0009-2614%2895%2900841-Q).
3. K. Raghavachari, G. W. Trucks, J. A. Pople, and M. Head-Gordon,
   “A fifth-order perturbation comparison of electron correlation theories,”
   *Chem. Phys. Lett.* **157**, 479–483 (1989).
   [DOI: 10.1016/S0009-2614(89)87395-6](https://doi.org/10.1016/S0009-2614%2889%2987395-6).
4. R. J. Bartlett and M. Musiał, “Coupled-cluster theory in quantum chemistry,”
   *Rev. Mod. Phys.* **79**, 291–352 (2007).
   [DOI: 10.1103/RevModPhys.79.291](https://doi.org/10.1103/RevModPhys.79.291).
5. Q. Sun et al., “Recent developments in the PySCF program package,”
   *J. Chem. Phys.* **153**, 024109 (2020).
   [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074).

6. M. Urban, J. Noga, S. J. Cole, and R. J. Bartlett,
   “Towards a full CCSDT model for electron correlation,”
   *J. Chem. Phys.* **83**, 4041–4046 (1985).
   [DOI: 10.1063/1.449067](https://doi.org/10.1063/1.449067).
   This is the method reference attached to CCSD+T(CCSD) in the
   [OpenMolcas manual](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/ccsdt.html).

The default triples correction is **conventional CCSD(T)**; the explicitly
selected Urban variant is **CCSD+T(CCSD)**. Solving a CCSD
Lambda equation does not turn this correction into Lambda-CCSD(T). CC3, CCSDT,
CCSDT(Q), renormalized corrections, pair CC and orbital-optimized CC are not
exported as implemented methods. Their definitions and citations must be added
with their own numerical validation rather than inferred from the names above.

The pinned OpenMolcas source review, method/program naming distinction and limits
of the cross-check are recorded in [OPENMOLCAS.md](OPENMOLCAS.md).

7. J. Gauss and J. F. Stanton, “Coupled-cluster calculations of nuclear magnetic
   resonance chemical shifts,” *J. Chem. Phys.* **103**, 3561 (1995).
   [DOI: 10.1063/1.470240](https://doi.org/10.1063/1.470240).
   Bibliographic identity checked against the
   [author's publication list](https://www.tc.uni-mainz.de/publikationen/publikationen-juergen-gauss/).
   The Table III attribution is inherited from the inspected PySCF source;
   this work does not claim an independent full-text audit of that table.
