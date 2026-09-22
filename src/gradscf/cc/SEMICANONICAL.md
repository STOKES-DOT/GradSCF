# General-reference triples through a tensor resolvent

The explicit option `orbital_basis="semicanonical"` extends real molecular
CCSD(T) to ROHF-based unrestricted amplitudes and to noncanonical occupied/virtual
orbital frames. It uses the general-reference spin-orbital formula attributed to
Watts, Gauss and Bartlett (1993), [doi:10.1063/1.464480](https://doi.org/10.1063/1.464480).
The working moments were adapted from
[PySCF 2.9.0 gccsd_t_slow](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/gccsd_t_slow.py);
source attribution and hashes are in [NOTICE.md](NOTICE.md).

This means unrestricted CCSD on the supplied ROHF determinant. It does not
introduce a separately spin-adapted ROCCSD formulation or imply equivalence to
every software package's ROCCSD(T) convention. Complex and spin-mixed GHF inputs
remain unsupported.

## API and backward compatibility

```python
mycc = cc.CCSD(mf).run()  # real HF, UHF or ROHF source
correction = mycc.ccsd_t(orbital_basis="semicanonical")
details = mycc.triples(orbital_basis="semicanonical")
```

The same keyword is accepted by `cc.evaluate_triples` and
`cc.triples_correction`. The default `orbital_basis="canonical"` retains the
previous strict canonical checks and streamed virtual-triples implementation.
There is no automatic algorithm switch or reference-orbital modification.
Only conventional `(T)` supports the new option; restricted Urban +T(CCSD)
continues to use its original canonical path.

## Forward equation

Let `o` and `v` denote active **spin** orbitals in the existing occupied-alpha,
occupied-beta, virtual-alpha, virtual-beta ordering. Frozen electrons contribute
to the Fock matrix before its active projection. Define the tensor operator

\[
\begin{aligned}
(\mathcal D X)_{ijkabc}={}&
\sum_l(F_{il}X_{ljkabc}+F_{jl}X_{ilkabc}+F_{kl}X_{ijlabc})\\
&-\sum_d(F_{ad}X_{ijkdbc}+F_{bd}X_{ijkadc}+F_{cd}X_{ijkabd}).
\end{aligned}
\]

The first sum is in the occupied block, the second in the virtual block.
In a basis that diagonalizes these blocks, `D` reduces to
`eps_i + eps_j + eps_k - eps_a - eps_b - eps_c`. Hence solving in the original
frame is algebraically equivalent to explicitly rotating the integrals and
amplitudes to semicanonical orbitals, dividing by these sums, and rotating back.

Using the fully antisymmetrized connected moment `W` and disconnected moment
`V` from the attributed formula,

\[
\mathcal D X=W,\qquad
E_{(T)}=\frac{1}{36}(W+V):X.
\]

`W` contains the two-electron-integral/T2 contractions. `V` includes **both**
the `T1 * <ij||ab>` contribution and `F_vo * T2`. The latter cannot be dropped
for a general reference: independent alpha/beta occupied–virtual Fock blocks
need not vanish for ROHF. No division by diagonal entries of an untransformed
noncanonical Fock matrix is substituted for the tensor inverse.

The resulting energy is invariant under separate active occupied and virtual
orthogonal rotations. Such rotations must preserve the frozen/active boundary.
They need not preserve an arbitrary choice of eigenvectors inside a degenerate
Fock block. Restricted t1/t2 are mapped to the existing spin-orbital representation
and evaluated with the same formula.

## Solver ownership and derivatives

`gradscf.solvers.solve_tensor_sum` owns the numerical factorization, true-residual
checks and implicit response. It accepts a tuple of symmetric factors and a
tensor RHS, without constructing the full Kronecker matrix. The triples code
passes three occupied factors and three negative virtual factors.

The numerical solve diagonalizes the small factors, but AD differentiates

\[
\mathcal D\,dX=dW-(d\mathcal D)X.
\]

The eigenvectors used by the numerical factorization are stopped *inside*
`custom_linear_solve`; the operator and RHS remain differentiable. Therefore
their response is retained without eigenvector gap denominators. A repeated
occupied/virtual eigenvalue is allowed as long as the tensor-sum inverse is
resolved. This is not the same as stopping the physical orbital dependence of
the correction. Right CC amplitudes retain their existing implicit response.
The validated complete CC contract remains first-order fixed-MO parameter AD,
not a nuclear gradient or a second-derivative claim for the entire CC method.

## Limits and diagnostics

- This opt-in path materializes six-index `W`, `V`, solution and work arrays.
  `max_triples_elements=2_000_000` limits `(nocc_spin*nvir_spin)**3` **per tensor**
  before moment construction. Two million float64 elements are 16 MB per array;
  this is not a peak process/device-memory bound. AD and simultaneous arrays
  require more memory. This path is a correctness reference, not a low-memory
  replacement for the default canonical streaming path.
- The common solver conservatively requires **all Cartesian eigenvalue sums**
  to be separated from zero, including repeated-index and spin-forbidden tensor
  entries. It may therefore reject an otherwise nonsingular antisymmetric
  physical subspace. It does not silently pseudoinvert or clip the physical
  correction. Restricting the inverse to exterior-power spaces is future work.
- `min_abs_denominator` in this mode refers to that Cartesian spectrum.
  `canonical_error` still reports the input active Fock off-diagonal maximum;
  it may be nonzero in a valid semicanonical calculation.
- `connected_component` is `W:X/36`; the existing `singles_component` field
  contains the whole applied disconnected term `V:X/36`, including `F_vo*T2`.
- The current CCSD residual, model identifier, amplitude symmetries and solve
  validity are checked. Empty triples spaces still require a matching valid
  CCSD state. Failed low-level corrections yield `valid=False`/NaN; facade
  requests raise. No denominator shifts are applied.

## Independent validation

Tests explicitly diagonalize active alpha/beta occupied and virtual blocks in
NumPy, rerun PySCF UCCSD in that frame, and use its optimized UCCSD(T) correction.
They cover ROHF OH/STO-3G (0.97 Angstrom), all-electron, frozen-core and
spin-dependent frozen spaces; random occupied/virtual rotations; restricted
H4 canonical agreement; and fixed-MO finite differences with off-diagonal Fock
perturbations. A synthetic interacting model tests exact within-spin orbital
degeneracy and complete converged-amplitude response. The shared solver has an
independent dense Kronecker-matrix oracle at exact degeneracy.

Run `examples/cc/semicanonical_triples.py` for a native GradSCF UHF/ROHF example.
Commands and measured results are recorded in [VALIDATION.md](VALIDATION.md).
