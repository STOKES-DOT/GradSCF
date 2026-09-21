# OpenMolcas reference review

The user-supplied [GitHub repository](https://github.com/Molcas/OpenMolcas) is a
mirror of the [official GitLab repository](https://gitlab.com/Molcas/OpenMolcas).
The source inspected here is pinned to commit
`4e52760a9bec07a7d253ee2fc5ac29fc5203bf27` (2026-09-17, “Parallel fix”).
The repository and inspected Fortran files carry LGPL-2.1 notices.

This review informed method definitions and control semantics. **No OpenMolcas
Fortran was copied or translated into the implementation.** The contractions
continue to build on the attributed PySCF/JAX code; see [NOTICE.md](NOTICE.md).
No OpenMolcas executable was built or run, so the tests below are not described
as numerical OpenMolcas comparisons.

## Inspected source

| File at the pinned commit | What was checked | SHA-256 |
| --- | --- | --- |
| [`src/ccsdt/ccsdt.F90`](https://github.com/Molcas/OpenMolcas/blob/4e52760a9bec07a7d253ee2fc5ac29fc5203bf27/src/ccsdt/ccsdt.F90) | Driver calls CCSD, then optionally CCT3 | `6d275344b16b35447da43edd833b77b47655a49193f75288fc311083b2660281` |
| [`src/cct3_util/t3reainput.F90`](https://github.com/Molcas/OpenMolcas/blob/4e52760a9bec07a7d253ee2fc5ac29fc5203bf27/src/cct3_util/t3reainput.F90) | Triples variants, denominator selection, separate triples shifts | `2be1a5996f71a8cd119ebba6f831a6f2099332b7976a1c56e49a593d40b092b1` |
| [`src/cct3_util/cct3.F90`](https://github.com/Molcas/OpenMolcas/blob/4e52760a9bec07a7d253ee2fc5ac29fc5203bf27/src/cct3_util/cct3.F90) | WT1 is conditional on typt3 > 1 and U*T2 on typt3 = 3 | `fc0481654961e37e24f6b9b300324c43d96498f4631d8189b255f5f3519f422d` |

## Consequences for GradSCF

1. The program name `CCSDT` does not establish fully iterative T1/T2/T3
   equations. The inspected driver and [manual](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/ccsdt.html)
   describe CCSD followed by noniterative triples.
2. `variant="ccsd+t(ccsd)"` selects the pure connected WT2 correction associated
   with Urban et al. (1985). `variant="ccsd(t)"` adds the singles term for the
   conventional canonical RHF correction of Raghavachari et al. (1989).
3. The off-diagonal-Fock U*T2 extension associated with the third OpenMolcas
   variant is not claimed here. Our corrections require canonical closed-shell
   active orbitals. The tiny off-diagonal noise allowed by the canonicality
   tolerance is not treated as an additional physical correction term.
4. `CCConfig.level_shift` shifts virtual levels only in the iteration
   preconditioner and initial guess. It does not change the CC residual, energy,
   Lambda equation, or physical triples denominators. The manual's distinction
   between iteration shifts and triples shifts motivated this split.
5. [CHCC](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/chcc.html)
   and [CHT3](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/cht3.html)
   describe Cholesky/RI and virtual-block approaches. Those storage/parallel
   algorithms have not been implemented by this review; full-MO storage remains
   a limitation of the current GradSCF implementation.

## Numerical validation strategy

- The Urban correction is checked independently as squared WT2 triple moments
  over Fock denominators in a PySCF FCI determinant basis.
- It is also checked against the connected-only limit of PySCF's slow triples
  routine, separately from the conventional CCSD(T) comparison.
- Both correction responses are checked along reconverged canonical paths;
  level-shift invariance is checked for CC energies and implicit derivatives.
- The one-particle density is a separate addition based on the CC Lagrangian.
  It is checked against PySCF and finite differences, not attributed to an
  OpenMolcas density routine.

Original method citations are in [REFERENCES.md](REFERENCES.md).
