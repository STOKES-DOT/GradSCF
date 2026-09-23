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
vectors and the full raw spectrum are forward diagnostics with stopped AD;
do not differentiate them or take a second derivative of the energy rule.
Complex selected roots produce NaN real outputs and preserve the raw complex
spectrum. Unconverged ground states/eigenpairs, unresolved gaps and excessive
conditioning invalidate energy derivatives (NaN), rather than changing the
method with a pseudoinverse. The response flag applies conservatively to the
whole requested root set. It checks gaps to the full computed spectrum,
including excluded roots. This first version does not attach a differentiable
non-Hermitian degenerate-subspace projector.

## Capacity and current limits

The solver materializes a bounded dense matrix from column-block actions:
`max_dense=256`, `block_size=16` by default. It is a small-system reference,
not an iterative matrix-free eigensolver. `max_intermediate_elements=20_000_000`
limits the input ERI element count before EOM intermediates are constructed;
it does not estimate peak memory or bound the earlier CC calculation.
No implicit fallback bypasses these limits.

UHF/ROHF, triplet/SF, higher-rank EOM methods, complex orbitals, transition
properties, vector/cluster response, GPU performance and nuclear gradients
remain outside this implementation. A generic Davidson/Arnoldi implementation
will be added to the shared solvers rather than maintained here.

Run [the native spectrum example](../../../../examples/cc/eom_ccsd.py) and
[the native response example](../../../../examples/cc/eom_response.py) on CPU.
Validation details and commands are in [VALIDATION.md](VALIDATION.md).

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
