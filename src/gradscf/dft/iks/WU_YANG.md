# Offline local-potential Wu-Yang inversion

The `wu_yang` entry prepares real RKS/collinear-UKS integer-occupation
references. It is separate from the existing finite-temperature full-matrix
inversion. It does not implement spinor inverse GKS or differentiate the inverse
map. Analytic/neural functional assessment at a fixed returned reference keeps
its existing parameter and density-derivative paths.

```python
potential = iks.Potential(grid, overlap, local_basis, nparam, max_memory=6000.)
target = iks.Target(density_target, overlap, nelec, energy=reference_total_energy)
inverse = iks.invert(target, potential, hcore, method='wu_yang')
if not inverse.valid:
    raise RuntimeError('Inspect density, moment, gradient and gap diagnostics')
reference = inverse.reference
```

`hcore` contains kinetic plus local nuclear attraction. A fixed local guide
may be added. A frozen Hartree-Fock Fock matrix is not an appropriate local-KS
base: its nonlocal exchange defines a different inverse problem.

## Objective and numerical representation

For F(b)=T+Vbase+sum_p b_p B_p and its ground-state density D(b), maximize
W=sum_i f_i epsilon_i-Tr[(Vbase+sum_p b_p B_p)Dtarget]. The implemented objective
omits the constant target kinetic term, so its reported value is not T_s.
It minimizes minus W plus alpha/2 times the squared norm of the observable
potential in the orthonormal AO frame.

The local PBS is integrated once into a bounded dense AO cache, with a memory
preflight against Potential.max_memory. The generic Potential grid producer
remains block based. Constant c*S directions are removed before normalizing
columns; a declared rank threshold then removes redundant/invisible AO
operators. `projection_rank` is the PBS-to-AO operator rank, not the density
response rank or a continuum uniqueness statement. The regularizer is an AO
operator norm, not an integral of the spatial potential gradient.

Optimization reuses the shared eager minimizer: L-BFGS with an optional actual
AD Hessian for small reference problems (<=256 reduced variables). The occupied
projector response permits internal occupied/virtual degeneracy with a nonzero
external gap. When energy changes are below floating-point resolution, the
shared driver can refine the gradient residual using its AD Jacobian and accepts
only reduced residual with no resolved energy worsening. Raw gradients remain
the convergence test. No tolerance is replaced by optimizer success.

## Acceptance and diagnostics

- `gradient_norm`: max absolute derivative of the regularized objective.
- `moment_error`: norm of the unregularized density moments in unit-AO-operator
  PBS coordinates, so a harmless rescaling of supplied basis functions does
  not alter acceptance. `raw_moment_error` preserves the original moment max.
- `density_error`: absolute spin-resolved grid L2 density error, bohr^-3/2.
- `gap`: minimum occupied-virtual gap, Hartree.
- `optimizer_evaluations` and `refinement_evaluations`: actual callback counts;
  `iterations` retains the main quasi-Newton iteration count.
- `valid`: requires actual gradient convergence, density/moment tolerances,
  a resolved gap and converged spectral solves. An invalid result keeps its
  diagnostic density/Fock/coefficients but has `reference=None`.

A correlated fractional AO1RDM is accepted as a target; the solver fits its real
space density, not its complete1RDM. Exact pure-state representability need not
hold in a finite Gaussian orbital/PBS pair. A nonzero regularizer also biases
the optimum. Report the density error and reference settings explicitly; do not
rename a merely converged optimizer an exact inverse.

## Verified first molecular check

`examples/iks/wu_yang_pbe.py` uses H2 at0.74 Angstrom, Cartesian def2-SVP,
grid level1 (4944points), known GradSCF PBE density, float64 and a local hcore
base. On jy_gpu_2/n005, Slurm110759, CPU4/8GiB, JAX0.10.0 CPU:

- inverse grid-density L2 error:7.515922384873537e-14;
- normalized moment error:3.2142194961661105e-13;
- regularized gradient max:3.2978119925237284e-16;
- gap:0.6490452061649511Ha;
- PBE total-energy reassessment error:-4.884981308350689e-15Ha;
- PBE orbital rotation residual squared:5.191049227425028e-21;
- inverse/source preparation elapsed:16.544858856126666s (one cold process,
  not a warm benchmark).

These are known-functional round-trip results, not CCSD-density teacher or
functional-discovery accuracy. Earlier strict-gradient failures are retained in
`artifacts/pbe-agent-loop-20261028/verification/roundtrip/`.

The new basis.cartesian_to_spherical map supports normalized-sp s/p/d shells.
It supplies an exact density embedding Dcart=T Dsph T.T for documented external
spherical data. It does not add a spherical SCF path. Source overlap must satisfy
Ssph=T.T Scart T before accepting an external matrix.
