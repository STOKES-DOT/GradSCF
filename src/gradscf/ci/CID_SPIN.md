# Doubles-only CI and total-spin diagnostics

`CID` and `UCID` select the reference determinant and **all double substitutions**.
They exclude singles. This is variational diagonalization in the selected space,
not CCD, not QCISD and not pair-only CI. Restricted/UHF/ROHF inputs, frozen
occupied/virtual orbitals and the existing real fixed-M_s conventions apply.
For a converged lowest root and fixed integrals, its energy cannot lie below
the lowest CISD energy in the enclosing space. CID is generally not size extensive.

The CID naming convention agrees with the
[Gaussian CID documentation](https://www.conflex.co.jp/gaussian_support/cid.php).
The determinant Hamiltonian and variational CI references remain those in
[REFERENCES.md](REFERENCES.md). No Gaussian executable was used for validation.

## Interfaces and excitation ranks

```python
myci = ci.CID(mf, nroots=2).run()  # also mf.CID().run()
ss, effective_multiplicity = myci.spin_square(root=0)
```

`CID` dispatches to the appropriate restricted/unrestricted integral path;
`UCID` explicitly requires an unrestricted or ROHF source. General `CI`/`UCI`
and `make_ci_space`/`make_uci_space` accept `excitation_ranks=(0,2)` for the same
space. `None` preserves the existing full hierarchy through `max_excitation`.
Ranks must be distinct nonnegative integers within the limit and include 0:
the solver uses the reference determinant to define the correlation energy.
The selector applies to **total alpha plus beta excitation rank**, so CID
contains alpha-alpha, alpha-beta and beta-beta doubles when allowed.

The determinant budget counts only the retained space, before construction;
CID is not implemented by allocating a larger CISD space and removing singles.
Only required per-spin excitation channels are generated. Changing ranks,
reference or frozen settings after a calculation invalidates property calls
until `kernel()` is rerun. Existing Hamiltonian construction, Davidson/dense
solvers and their forward/backward rules are reused unchanged.

## Total spin in distinct orbital frames

For orthonormal spatial alpha/beta MO frames define

\[
O_{ps}=\langle p_\alpha|s_\beta\rangle
       =(C_\alpha^\mathsf{T}S_{\rm AO}C_\beta)_{ps},
\qquad
\Gamma^{\alpha\beta}_{pqrs}
=\langle a^\dagger_{p\alpha}a^\dagger_{r\beta}a_{s\beta}a_{q\alpha}\rangle.
\]

In a fixed `(Nalpha,Nbeta)` sector, the identity
`S^2 = Sz^2 + (S+ S- + S- S+)/2` gives

\[
\langle S^2\rangle
=\frac{(N_\alpha-N_\beta)^2}{4}+\frac{N_\alpha+N_\beta}{2}
-\sum_{pqrs}\Gamma^{\alpha\beta}_{pqrs}O_{ps}O_{qr}.
\]

The implementation contracts the existing normalized fermionic 2-RDM, including
frozen electrons. The identity and index convention are cross-checked against
[PySCF's spin operator](https://pyscf.org/_modules/pyscf/fci/spin_op.html), which
is used as an independent test oracle, not a runtime dependency.

`ci.spin_square(coefficients, space, overlap_ab=...)` returns
`(<S^2>, sqrt(1+4*<S^2>))`, using hbar=1 (S^2 in units of hbar squared).
It accepts unnormalized real vectors under the same
normalization policy as the density functions; zero or nonfinite states produce
NaNs. Singlet/triplet/quintet eigenstates give `(0,1)`, `(2,3)`, `(6,5)`.
The second quantity is an **effective multiplicity**, not a rounded spin label
or proof that the state is a spin eigenfunction. The diagnostic neither projects
spin nor changes the roots chosen by the energy solver.
In particular, a spin-pure ROHF reference alone does not make the selected
truncated CI space spin adapted; its correlated roots can also mix total spin.

- `CISpace` has one common spatial frame and uses `O=I`; an explicit overlap
  override is rejected for that representation.
- `UCISpace` requires an explicit real `(nmo,nmo)` `overlap_ab`. The one-/two-body
  Hamiltonian alone does not supply cross-spin orbital overlaps.
- An SCF-backed facade obtains `C_alpha.T @ S @ C_beta` for UHF and identity for
  ROHF's common orthonormal spatial orbitals. An explicit `UnrestrictedReference`
  must supply `overlap_ab` to `.spin_square()`; it is never silently set to identity.
- The two MO frames need not span identical spatial subspaces. `O` then need
  not be an orthogonal matrix. Frame orthonormality and consistency of the
  supplied overlaps with the orbitals are the caller's responsibility.

## Differentiation and limits

The functional diagnostic supports JIT and AD with respect to coefficients and
overlaps. To differentiate a converged CI state's diagnostic with respect to
MO integrals, use `gradient_mode="implicit_eigenvector"` on an isolated root.
The default energy-only mode intentionally stops coefficient response.
The eager facade is not a differentiable SCF/nuclear-coordinate interface.

The current diagnostic uses a full MO 2-RDM and inherits its connection/storage
limits; it is a small-system reference implementation, not a large-CI spin-action
algorithm. It applies to determinant `CI`/`CID`/`CISD`/higher-rank results, including
their unrestricted counterparts. `CIS`, `UCIS` and `CIS_D` return different
amplitude/result representations and are not silently converted by this API.
Automatic spin selection, penalties, spin variance, transition-spin matrices,
complex/GHF states and degenerate-root tracking are not added here.

See [the native example](../../../examples/ci/cid_spin.py) and
[validation results](VALIDATION.md).
