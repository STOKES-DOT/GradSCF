# Integral implementation references and boundaries

Reviewed2026-10-26. These sources guide layout/ownership choices; they are not
performance measurements for this worktree.

* **libcint6.1.3**: the directly downloaded commit
  [c72d306](https://github.com/sunqm/libcint/tree/c72d30662ba08048940c9000cc352a220c867712)
  is the actual Gaussian/Coulomb kernel source. The local archive and all45
  selected vendor files were hashed and compared bytewise; see
  `upstream_verification.json`. New products live in GradSCF's private adapter,
  leaving vendor files unchanged. Public ATM/BAS/ENV layouts, generated intor
  signatures, Cartesian/spherical order and shell-cache rules are retained.
* **PySCF**: pinned C filling/ECP source and independent numerical references
  are retained with licenses. Python `mol.intor` is not a production bridge.
  Shared-input oracles remain distinct from independent GradSCF SCF workflows.
* **Psi4 libmints**: its public
  [two-body source](https://github.com/psi4/psi4/blob/master/psi4/src/psi4/libmints/twobody.cc)
  exposes shell-block computation and separate derivative entry points. The
  relevant design lesson is explicit shell output ownership and consumption
  at the requested block boundary. No Psi4/Libint production runtime was added.
* **ORCA**: the official
  [manual and its section index](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/basisset.html)
  separate integral handling and the SHARK task driver from basis/auxiliary
  specifications.
  This is documentation review, not inspection or copying of an available
  full source distribution. It supports keeping representation/task policy
  separate from the sole Gaussian kernel and preserving auxiliary-basis roles.
* **ABACUS**: the public
  [NAO module](https://github.com/deepmodeling/abacus-develop/tree/develop/source/source_basis/module_nao)
  provides an architecture comparison for numerical atomic orbitals. Those
  numerical-orbital operations must not be conflated with a second analytic
  Gaussian engine; none of its numerical kernels were transplanted here.
* **FHI-aims**: the official
  [code-access page](https://fhi-aims.org/get-the-code) describes licensed access.
  No claim of full-source inspection or code reuse is made. Its numerical-AO
  setting is a separate method context from GradSCF's libcint Gaussian kernels.

The concrete parallel implementation uses worker-private copies of mutable
libcint tables and shell scratch, a shared read-only cotangent, and a fixed
reduction order. Derivative products do not inherit primal coefficient masks:
an individual zero coefficient can have a nonzero derivative. Raw coefficient
and primitive/contraction normalization chains remain in JAX. These choices
are verified by FD/transpose/HVP and thread-parity tests, rather than inferred
from another program's language or package name.

Caches separate immutable topology/program identity from current physical
leaves. Nuclear positions, independent AO centers, coefficients, exponents,
densities and model parameters must not be converted into static cache keys
or reused as converged solutions. Existing fixed-geometry RI caches retain
their signature checks; changes of geometry or exponents are rejected.
