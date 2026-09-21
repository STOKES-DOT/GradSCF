# Restricted single-reference configuration interaction

`gradscf.ci` implements real, molecular, closed-shell-reference CI with JAX.
PySCF is used in the tests as an independent reference, not by the solver.
Numerical eigensolves and their response rules are owned by
[`gradscf.solvers`](../solvers/README.md); CI constructs the Hamiltonian action
and maps the common solver results to electronic-structure quantities.

## Methods and interfaces

| Interface | Space and energy convention |
| --- | --- |
| `CIS(mf, singlet=True/False)` | Spin-adapted singles; excitation energies relative to HF |
| `CIS_D(mf)` | Canonical RHF singlet CIS(D); corrected excitation energies |
| `CISD(mf)` | Reference plus all singles and doubles, fixed M_s = 0 |
| `CISDT(mf)` | Reference plus all excitations through triples, fixed M_s = 0 |
| `CISDTQ(mf)` | Reference plus all excitations through quadruples, fixed M_s = 0 |
| `CI(mf, max_excitation=k)` | General rank-truncated space, fixed M_s = 0 |

The determinant CI solvers fix N-alpha = N-beta. They do **not** select total
spin S: higher roots can be singlets, triplets, or higher-spin states. Only CIS
currently exposes explicit singlet/triplet adaptation. This differs from the
spin-adapted restricted CISD amplitudes in PySCF; raw CI coefficient arrays are
not interchangeable. Truncated CI is generally not size extensive.

```python
from gradscf import gto, dft, ci

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
mf = dft.RKS(mol, xc="hf").run()
myci = ci.CISD(mf).run()  # equivalently mf.CISD().run()
print(myci.e_tot, myci.e_corr, myci.converged)

singles = ci.CIS(mf, nroots=1, singlet=True).run()
corrected = ci.CIS_D(mf, nroots=1).run()
print(singles.e, corrected.e, corrected.correction)
```

All energies are in Hartree. `CI.kernel()` returns `(e_corr, coefficients)`;
`CIS.kernel()` and `CIS_D.kernel()` return `(excitation_energies, amplitudes)`.
`.run()` returns the solver object. Scalar CI facade attributes are used for one
root; `CIResult` always keeps an explicit root axis, with coefficient columns of
shape `(ndeterminants, nroots)`. CIS amplitudes have shape `(nroots, nocc, nvir)`
in the active space and unit spatial norm. PySCF restricted TDA amplitudes have
half that squared norm; the legacy GradSCF CIS(D) adapter converts explicitly.

`frozen=2` freezes the first two doubly occupied spatial orbitals.
`frozen=[0, 1, 8]` freezes arbitrary occupied or virtual spatial orbitals.
Frozen electrons remain in all determinants: their core energy and interactions
with active electrons are retained rather than removed by array slicing.

## Functional and differentiable API

Construct determinant topology outside `jit`, then close over it:

```python
import jax
from gradscf.ci import CIConfig, make_ci_space, solve_ci

space = make_ci_space(nmo=4, nocc=2, max_excitation=3)
config = CIConfig(conv_tol=1e-10)

@jax.jit
def energy(h1_mo, eri_mo):
    return solve_ci(h1_mo, eri_mo, space, config=config).total_energies[0]

dh = jax.grad(energy, argnums=0)  # callable; supply real MO integrals
```

MO orbitals must be orthonormal and occupied orbitals must precede virtual ones.
`eri[p,q,r,s]` uses chemists' notation `(pq|rs)`. The one-electron Hamiltonian
must be symmetric, and the real ERIs must obey their usual permutation
symmetries. The low-level kernels check shapes and dtypes; physical consistency
of supplied integrals and the reference is the caller's responsibility.

For explicit inputs to a facade use
`CIReference(h1_mo, eri_mo, nocc, nuclear_repulsion=..., mo_energy=...)`.
For AO inputs, `ci.integrals.transform_integrals` supports full ERIs, an s4
AO-pair matrix, or density-fitting factors with shape `(naux, nao, nao)`.
The SCF facade adapter is eager and accepts converged GradSCF `RKS(xc="hf")`;
use functional kernels inside `jax.jit`, `jax.grad`, and `jax.jvp`.

The default `gradient_mode="eigenvalue_only"` stops the iterative Ritz vectors
and differentiates their Rayleigh energies. At convergence and for an isolated
root, dE = c.T (dH) c. Coefficient derivatives are disabled in this mode.
`gradient_mode="implicit_eigenvector"` additionally solves the constrained
eigenvector response using the existing TD-SCF adjoint implementation. This
mode is necessary for coefficient-dependent objectives, including CIS(D).
These rules implement **first-order** derivatives of converged, nondegenerate
roots. Higher derivatives and differentiable root tracking at degeneracies
are not supported contracts.

Unconverged roots remain available for diagnostics; `residual_norms` and
`converged` are per-root arrays. Their energy derivatives are NaN. Check all
requested roots before differentiation. CI AD is with respect to its numerical
inputs; complete nuclear gradients also require SCF orbital response and the
chosen integral backend's derivative support. No automatic backend fallback is
introduced here.

## CIS(D) theory and validation

The singlet correction is the unscaled Head-Gordon CIS(D) expression:

```text
delta omega = <CIS|V|U2 HF> + <CIS|V|T2 U1 HF> - E_MP2
```

`U2` uses the excitation-energy-shifted doubles denominators; `T2` uses MP2
denominators. The second term contains disconnected triples. This is not
variational CISD, and is not merely an external-doubles EN2 correction.
The formula and subtraction are described in the
[Q-Chem manual, equations 7.38–7.40](https://manual.q-chem.com/5.1/sect-excorr.html).
The implementation reuses `tddft.cisd.restricted_cisd_second_order_correction`.
The independent test constructs the doubles and disconnected triples explicitly
in PySCF's FCI determinant basis and verifies all three contributions together.

`cis_d_correction(eri_mo, mo_energy, singles, nocc=..., frozen=...)` exposes
the pure JAX correction. Supply canonical RHF orbital energies and singlet CIS
amplitudes. Its result contains original energies, corrections, corrected
energies, the minimum absolute denominator per root, and `valid` flags.
Denominators below `denominator_tol` (default 1e-10 Hartree) produce invalid/NaN
results; no shift or clipping changes the theory. Correction gradients from an
energy-only CIS result are NaN to avoid returning incomplete derivatives.
The `CIS_D` facade enables amplitude response automatically and checks that
the Fock matrix is diagonal and consistent with the supplied orbital energies.

## Scope and computational limits

- The Hamiltonian is applied through stored Slater–Condon connections; Davidson
  does not construct a dense determinant Hamiltonian. Static connections are
  cached on the host, while integral contractions and vector actions use JAX.
- This generic implementation targets small reference calculations. Full MO
  ERIs still require O(nmo^4) storage, and the connection table can be large.
  It is not an optimized production CISD/FCI engine.
- The default space limit is 5000 determinants (configurable); connection
  construction stops at 1,000,000 stored diagonal/undirected connections.
  Dense verification is limited to 2048 determinants. Reducing the orbital
  space or excitation rank is preferable to simply increasing these limits.
- UHF/ROHF/GHF, complex orbitals, periodic CI, RDM/property APIs, and automatic
  spin selection for general CI roots are not implemented in this first stage.
- Named `CISD(T)` / `CISDT(Q)` corrections are not exported. They need a specific
  perturbative definition and reference implementation; these names are not
  inferred from the coupled-cluster hierarchy.

## Reproduce the checks

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/ci
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python examples/ci/restricted_ci.py
```

Tests use H2 (0.74 Angstrom), asymmetric linear H4 (z = 0, 0.8, 1.9, 3.1
Angstrom), and LiH (1.6 Angstrom), with STO-3G, RHF, CPU, and float64. Energy
comparisons to PySCF use absolute tolerances of 2e-9 Hartree (1e-8 for the native
end-to-end H2 example); Hamiltonian elements and FCI limits use 2e-12 Hartree.
Directional finite differences use a 1e-4 parameter step with approximately
2e-7 derivative tolerance. These are validation tolerances, not error estimates
for the electronic-structure approximation.

Additional references: [PySCF CI API](https://pyscf.org/user/ci.html),
[Psi4 determinant CI theory and spin conventions](https://psicode.org/psi4manual/master/detci.html).
