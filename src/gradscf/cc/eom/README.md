# Closed-shell EOM-CCSD: EE, IP and EA

This first implementation supports real molecular restricted CCSD references,
**EE singlets and IP/EA doublets**. Numerical diagonalization belongs to
`gradscf.solvers.solve_nonhermitian`; this package builds the physical actions.
PySCF is used only for independent comparisons, not at runtime.

## Public API

```python
from gradscf import cc

mycc = cc.CCSD(mf).run()  # mf is a converged GradSCF restricted HF object
excited = cc.EOMEE(mycc, nroots=3).run()
ionized = mycc.EOMIP(nroots=3).run()
attached = mycc.EOMEA(nroots=3).run()
print(excited.e, ionized.e, attached.e)  # Hartree, each shape (nroots,)
```

Convenience methods `mycc.eomee_ccsd_singlet()`, `mycc.ipccsd()` and
`mycc.eaccsd()` return `(energy, vector)` for one root or `(energies, vectors)`
for multiple roots, with vectors as rows in the latter case. Result-object
`right_vectors` and `left_vectors` always store roots in **columns**.
`amplitudes(root)` unpacks the right vector. There is intentionally no generic
all-spin EE dispatcher. Recomputing the source or changing frozen spaces,
amplitudes, or settings invalidates the eager snapshot; use `check_source()`
before reusing one after mutation.

| Sector | Operators retained | Energy convention | Packed dimension |
| --- | --- | --- | --- |
| EE singlet | 1p1h + 2p2h | E_k(N) - E_0(N) | ov + ov(ov+1)/2 |
| IP doublet | 1h + 2h1p | E_k(N-1) - E_0(N) | o + o²v |
| EA doublet | 1p + 2p1h | E_k(N+1) - E_0(N) | v + ov² |

Here o/v count active spatial orbitals. The conventional electron affinity is
**minus the EA eigenvalue**. Negative roots are retained. Frozen occupied and
virtual orbitals use the existing ground-state active-space definition.

## Equations and coordinates

For T = T1 + T2, the similarity-transformed Hamiltonian is
Hbar = exp(-T) H exp(T). The ground amplitudes obey F(T;h,g)=0. EE uses the
Jacobian A = dF/dT of the existing physical CCSD residual, which is equivalent
to the projected connected commutator at a converged reference. IP/EA use the
appropriate particle-number projections of Hbar - E_CC. Charged-sector
contractions are adapted from PySCF 2.9.0; see [source attribution](../NOTICE.md).

The EE independent doubles coordinates inherit `cc.AmplitudeSpace`: the
symmetric (ia,jb) matrix has sqrt(2)-weighted off-diagonal entries. PySCF's
unweighted packing differs by a similarity transformation. Tests unpack and
repack every probe and action before comparing. IP/EA use flattened spatial
r1/r2 arrays, without imposing an extra doubles antisymmetry.

The eigenproblem is non-Hermitian:

```
A R_k = omega_k R_k
A.T L_k = omega_k L_k
L.T R = I
```

Left vectors are duals in these same packed coordinates. They are neither
ordinary right-vector transposes nor the ground-state Lambda amplitudes.
They alone do not define transition moments or oscillator strengths.
Right vectors have unit Euclidean norm; the left norm then estimates the
simple eigenvalue condition number. Both eigen-equation residuals and the
biorthogonality error are reported.

## First-order automatic differentiation

For an isolated real root with L_k.T R_k = 1,

```
d omega_k = L_k.T (dA) R_k
```

The numerical eigenvectors are stopped when attaching this first-order energy
rule. To include the change in the converged CC amplitudes, solve CC **inside**
the differentiated function:

```python
import jax
from gradscf import cc

cc_cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
eom_cfg = cc.EOMConfig(sector="ip", nroots=1)

def energy(t):
    ht = h + t * perturbation
    ground = cc.run_cc(ht, eri, nocc=nocc, config=cc_cfg)
    result = cc.run_eom(ht, eri, ground, nocc=nocc,
                        config=eom_cfg, cc_config=cc_cfg)
    return result.energies[0]

value, derivative = jax.jit(jax.value_and_grad(energy))(0.0)
```

Here the derivative is with respect to the explicit scalar perturbation t in
the fixed MO frame. `run_cc` supplies implicit T response. Holding a previously
computed ground result constant gives only a fixed-T partial derivative.
This is not a nuclear-coordinate gradient or differentiation through HF.

Only **first-order energies** are supported by this interface. Left/right
vectors and the raw spectrum are forward diagnostics with stopped AD;
do not differentiate them or take a second derivative of the energy rule.
Complex selected roots produce NaN real outputs and preserve the raw complex
spectrum. Unconverged ground states/eigenpairs, unresolved gaps and excessive
conditioning invalidate energy derivatives (NaN), rather than changing the
method with a pseudoinverse. The response flag applies conservatively to the
whole requested root set. Dense solving checks the full computed spectrum.
Davidson checks the projected spectrum and converged guard roots, subject to
the spectral-completeness limitation below. This first version does not attach a differentiable
non-Hermitian degenerate-subspace projector.

## Capacity and current limits

The default `solver="dense"` materializes a bounded reference matrix:
`max_dense=256`, `block_size=16`. Select the shared iterative implementation
explicitly for larger EOM spaces:

```python
ee = cc.EOMEE(mycc, nroots=2, solver="davidson", max_space=40,
              max_cycle=150, guard_roots=1, seed=0).run()
print(ee.e, ee.result.iterations, ee.result.restarts)
print(ee.result.spectrum_complete, ee.result.spectral_gaps)
```

Davidson stores a real orthonormal search basis, expands with both right and
left residuals and restarts with both Ritz spaces. A stalled diagonal
preconditioner falls back to raw residual expansion. EOM provides orbital-energy
difference preconditioners and the physical action; its transpose is supplied
by JAX. No physical EOM matrix is assembled, including during energy AD.
Numerical storage is O(n*m + m²), m <= `max_space`. There is no dense fallback.
For a truncated search space, `max_space` must be at least
`4*(nroots+guard_roots)+2` to retain complex directions and expand after restart.
The extra guard roots must also converge before `converged` becomes true.

**Finite Ritz sampling does not certify the full spectral order or isolation.**
`spectrum_complete` is true only when the search basis spans the physical space
(or for the dense reference). Otherwise `spectral_gaps` and root ordering are
estimates. `response_valid=True` means that requested/guard residuals, observed
gaps and conditioning passed; it is conditional on the selected branch really
being isolated in the full spectrum. A small residual cannot exclude an unseen
invariant sector. Default diagonal guesses carry small independent full-support perturbations
to avoid immediate termination inside an exact invariant sector; independent
seeded vectors provide additional exploration. Explicit `initial_vectors` in
the shared solver are preserved. These choices improve exploration but are
not a proof. Validate sensitive crossings/near-degeneracies with larger spaces,
more guard roots/independent starts, and a dense oracle when feasible.

`raw_eigenvalues` holds the full dense spectrum or the final projected spectrum;
inactive padded entries are NaN. `subspace_dimension`, `iterations`, `restarts`
and `guard_residual_norms` describe the actual numerical solve. Guard residuals
are the maximum of unit-right and unit-left residual norms.

`max_intermediate_elements=20_000_000` still limits the input ERI element count
before EOM intermediates; it does not estimate peak memory or bound the earlier
CC calculation. Removing the EOM matrix bottleneck does not make integrals or
CC intermediates low-memory algorithms.

UHF/ROHF, triplet/SF, higher-rank EOM methods, complex orbitals, transition
properties, vector/cluster response, GPU performance and nuclear gradients
remain outside this implementation.

Run [the dense native spectrum example](../../../../examples/cc/eom_ccsd.py),
[the response example](../../../../examples/cc/eom_response.py) and
[the iterative 6-31G example](../../../../examples/cc/eom_iterative.py) on CPU.
See [initial validation](VALIDATION.md) and
[iterative validation](ITERATIVE_VALIDATION.md), and the
[eight-molecule study](MOLECULAR_VALIDATION.md). The latter records a higher-energy
native N2 SCF branch and a seed-dependent rank loss in CO's degenerate EE left
vectors. Convergence checks reject the CO state output; changing search settings
is a workaround, not a repair of the degenerate-subspace representation.

## Method references

- J. F. Stanton and R. J. Bartlett, *The equation of motion coupled-cluster
  method. A systematic biorthogonal approach to molecular excitation energies,
  transition probabilities, and excited state properties*, J. Chem. Phys.
  **98**, 7029–7039 (1993), [doi:10.1063/1.464746](https://doi.org/10.1063/1.464746).
- M. Nooijen and R. J. Bartlett, *Equation of motion coupled cluster method for
  electron attachment*, J. Chem. Phys. **102**, 3629–3647 (1995),
  [doi:10.1063/1.468592](https://doi.org/10.1063/1.468592).
- M. Nooijen and J. G. Snijders, *Second order many-body perturbation
  approximations to the coupled cluster Green's function*, J. Chem. Phys.
  **102**, 1681–1688 (1995), [doi:10.1063/1.468900](https://doi.org/10.1063/1.468900).
  The upstream IP implementation cites its Eqs. (8)–(9); the contractions here
  use the full converged CCSD amplitudes, not an MP2 truncation.

EE/EA bibliographic metadata checked against publisher-deposited Crossref
records; IP metadata checked against the authors' university record.
Scientific attribution is separate from the Apache source-code attribution.
