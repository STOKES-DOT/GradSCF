# CI/CC forward capability comparison and implementation stages

Audit date: 2026-09-22. GradSCF baseline: `52d2d93`; the ground-state additions
below belong to `feat/ci-cc-forward-parity`. This is a capability comparison,
not a speed benchmark or a claim of complete software equivalence.

The comparison pins Gaussian 16 documentation, Q-Chem **6.3**, ORCA **6.1**,
and the PySCF user documentation, with **PySCF 2.9.0** as the local numerical
oracle. These are explicit reference versions, not a claim that each is the
latest release. Gaussian's main keyword pages returned HTTP 403; the links below
use the Gaussian 16 manual translation published by its vendor CONFLEX.
Commercial packages were not executed locally.

## What the reference packages provide

| Package | CI and conventional ground-state CC | Beyond the basic ground-state energy |
| --- | --- | --- |
| Gaussian 16 | CIS/CIS(D), CID/CISD, CCD/CCSD/CCSD(T), QCISD/QCISD(T); RO CCSD/(T) energy options | EOM-CCSD, correlated densities/properties, frozen-core and checkpoint/restart controls; derivative availability depends on method |
| Q-Chem 6.3 | CCSD, iterative CCSDT, CC2, QCISD, OD/QCCD and noniterative corrections; EOM/CI family | EE/IP/EA/SF and double attachment/detachment sectors, RI/CD-CC, CC properties/gradients and frozen-natural-orbital controls |
| ORCA 6.1 | Canonical CISD/QCISD/CCSD/(T) and CEPA/CPF families; DLPNO and F12 variants | EOM/STEOM, unrelaxed correlated properties, local-correlation controls; high-order methods also appear through a separately documented MRCC interface |
| PySCF | R/U/G-CISD, separate FCI engine, R/U/G-CCSD/(T), restricted QCISD/(T) implementation | Lambda, MO/AO 1/2-RDMs, frozen orbitals, EOM EE/IP/EA/SF, DF/direct/out-of-core CC paths, selected nuclear gradients |

Sources for each row:

- Gaussian: [CIS/CIS(D)](https://www.conflex.co.jp/gaussian_support/cis.php),
  [CID/CISD](https://www.conflex.co.jp/gaussian_support/cid.php),
  [CCD/CCSD/(T)](https://www.conflex.co.jp/gaussian_support/cc.php),
  [QCI](https://www.conflex.co.jp/gaussian_support/qci.php),
  [EOM-CCSD](https://www.conflex.co.jp/gaussian_support/eom.php).
- Q-Chem: [ground-state methods](https://manual.q-chem.com/6.3/topic_ccsd.html),
  [triples/correction controls](https://manual.q-chem.com/6.3/Ch6.S13.SS4.html),
  [EOM/CI suite and properties](https://manual.q-chem.com/6.3/topic_eomcc.html).
- ORCA: [MDCI methods and local variants](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html),
  [EOM and links to related modules](https://www.faccts.de/docs/orca/6.1/manual/contents/spectroscopyproperties/eom.html).
- PySCF: [CI](https://pyscf.org/user/ci.html), [CC](https://pyscf.org/user/cc.html),
  [QCISD source](https://pyscf.org/_modules/pyscf/cc/qcisd.html).

The entries describe families, not every combination of spin reference,
gradient, density, local approximation and excited-state sector. In particular,
ORCA's documented open-shell DLPNO path targets high-spin references and has
restrictions for broken-symmetry systems. A native implementation and an external
program interface are different capabilities. A perturbative triples correction
is not iterative CCSDT. Named CISD(T)/CISDT(Q) corrections are not inferred from
the CC hierarchy: each needs its own definition and numerical reference.

## GradSCF audit and the first implementation stage

| Capability | Baseline | This stage |
| --- | --- | --- |
| Restricted CIS, singlet CIS(D) | Available | Unchanged |
| UHF/ROHF determinant CI through arbitrary rank | Available, small spaces | Add normalized 1/2-RDMs and per-root facade access |
| CID/UCID | Missing | Reference plus doubles-only spaces, with retained-space budget checks |
| General CI spin diagnostics/selection | Fixed M_s only | Add <S^2> and effective multiplicity using actual cross-spin overlaps; total-spin selection remains missing |
| Restricted CCS/CCD/CCSD/CC2/LCCD/LCCSD | Available | Preserve their current definitions |
| Restricted QCISD/QCISD(T) | Missing | Add quadratic residual/energy, canonical QCI triples, model adjoint/1-RDM and first-order response; U/ROHF QCI remains missing |
| UHF/ROHF UCCSD/UCCD | Available | Add explicit Lambda and spin-resolved 1-RDMs |
| Restricted conventional (T), Urban +T(CCSD) | Available | Preserve existing paths |
| Canonical UHF CCSD(T) | Missing | Add streamed spin-orbital triples, diagnostics and amplitude response |
| Real ROHF/noncanonical triples | Missing | Opt-in semicanonical-equivalent tensor resolvent, including F_vo*T2; capacity and conservative spectrum guards |
| R/U CCSD/CCD 2-RDM | Missing | Add true fermionic density with Lambda and frozen-core restoration |
| CI/CC property state freshness | CC guarded; no CI density interface | Share reference fingerprints; reject stale CI densities |
| Optimized tensor CI / low-memory CC | Missing | Still missing; full MO/spin tensors remain in use |
| Restricted EOM-EE singlet / IP / EA | Bounded dense reference | Left/right states and isolated-energy first-order AD; see [scope](eom/README.md) |
| Other EOM sectors/properties, iterative triples/quadruples, local/F12 | Missing | Subsequent stages; no placeholder APIs |

The user selected conventional single-reference ground states first. This stage
adds forward observables while preserving the existing first-order AD contract.
It is not full parity with all four packages, and it does not establish large
molecule performance. In particular, a DF integral adapter that reconstructs a
full MO tensor does not constitute a low-memory DF-CC implementation.

## Mathematical and API contract

CI properties use fermionic creation/annihilation connections and the normalized
CI vector. Frozen occupied electrons remain present. Restricted densities are
spin summed; UCI returns `(dm1a, dm1b)` and `(dm2aa, dm2ab, dm2bb)` in their own
orbital frames. `CI.make_rdm12(root=0)` selects one converged root. The low-level
`ci.make_rdm12(coefficients, space)` supports JIT and coefficient AD. Integral
response additionally requires the implicit-eigenvector CI mode and an isolated
root; energy-only coefficients have intentionally stopped response.

CC uses the stationary Lagrangian `L = E(T) + lambda^T R(T)`. The shared solver
solves `R_T^T lambda = -E_T`; no Lambda iteration is maintained by a CC submodule.
Unrestricted independent doubles use coordinates `x2 = 2*t2`. Hence unpacking
the dual requires `l2 = 4*unpack(lambda)_2` for the conventional `.25*l2.R2`
full-spin pairing. A CC density includes this left state and is not simply a
right-state expectation value.

Both density families use

```text
dm1[p,q]     = <a_q^+ a_p>
dm2[p,q,r,s] = <a_p^+ a_r^+ a_s a_q>
E = sum(h[p,q]*dm1[q,p]) + 1/2 sum(eri[p,q,r,s]*dm2[p,q,r,s]) + E_nuc
```

The unrestricted two-electron contraction weights are `(1/2,1,1/2)` for
`(aa,ab,bb)`. CC returns the real Hermitian part appropriate to real observables.
The CCSD 2-RDM also covers its CCS/CCD cluster restrictions. CC2/linearized-model
2-RDMs and CCSD(T) densities are explicitly outside this implementation.
All these densities are orbital-unrelaxed; nuclear gradients require additional
orbital/integral response.

Default canonical UCCSD(T) uses distinct virtual spin-orbital triples, occupied-cube intermediates,
physical unshifted denominators and explicit spin/Pauli masks. It validates the
current CCSD residual, active Fock canonicality and amplitude antisymmetry, even
when the triples space is empty. Invalid low-level corrections return NaNs with
`valid=False`; the facade raises. The documented default loop limit is 20,000
virtual triples. No full T3 tensor, PySCF runtime, denominator clipping that
changes the valid theory, or independent solver implementation is introduced.

The subsequent `orbital_basis="semicanonical"` increment adds a bounded
six-index reference path for real ROHF/noncanonical input. It uses the
general-reference `F_vo*T2` term and a shared tensor-sum inverse, avoiding
eigenvector derivatives at degenerate Fock eigenvalues. Its memory use and
conservative Cartesian-spectrum policy differ from the default streamed
canonical path; see [SEMICANONICAL.md](SEMICANONICAL.md).

## Next stages

1. Finish conventional ground-state breadth: unrestricted/ROHF QCI, exact CI spin
   selection, AO densities/one-electron property conveniences, robust checkpoint
   restart and correlated-gradient boundaries. Add each method with an oracle.
2. Extend the initial restricted EE/IP/EA reference with iterative solving, SF,
   transition densities and oscillator strengths. The nonsymmetric eigensolver and its response belong in
   `gradscf.solvers`, with explicit treatment of root selection and degeneracy.
3. Establish practical scaling: tensor CISD actions, spin-block CC intermediates,
   genuine DF/CD contractions, memory budgets, blocking/out-of-core storage and
   representative basis-size benchmarks.
4. Add CC3/iterative CCSDT and higher corrections with independent equations and
   tests. Local PNO/DLPNO, F12, spin-mixed/complex and multireference approaches
   are separate projects, not aliases of existing methods.

## Reproduction

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/ci tests/cc tests/solvers/test_nonlinear.py
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/cc/open_shell_properties.py
```

The tests set JAX float64, and use PySCF only as an oracle. Coverage includes
arbitrary determinant vectors, RHF H4/STO-3G, UHF H3/STO-3G and OH/6-31G,
frozen occupied/virtual orbitals, empty spin/amplitude spaces, contraction and
energy identities, canonical restricted/unrestricted (T) agreement, JIT,
reconverged finite differences and stale/invalid states. Numerical tolerances
are asserted in the tests. Executed results are recorded in [VALIDATION.md](VALIDATION.md).
