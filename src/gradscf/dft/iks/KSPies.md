# Standard molecular inversion: KSPies reference and GradSCF implementation

Algorithm reference: [KSPies](https://github.com/ssnam92/KSPies), pinned at
`cd99703ae588d3e081e97c8caec5c2aeff66d546`, Apache-2.0. The upstream Python
implementation is used only by the independent comparison example. GradSCF
runtime code imports its own integral, spectral, nonlinear and XC modules.

## Finite-penalty ZMP

Let D_tar be a target AO density in the same overlap metric S as the orbital
basis, and J[D] a fixed linear Coulomb action. The target may be a correlated
non-idempotent1RDM; matching its real-space density does not require matching
all matrix elements. A recovered KS density is built from integer-occupied
orbitals C with C^T S C=I.

For a fixed local base Hamiltonian F0 and restricted spin-summed density D,
consider

    E_lambda[D] = Tr(F0 D) + lambda/2 Tr((D-D_tar) J[D-D_tar]).

Self-adjointness of the Coulomb operator gives the stationary matrix

    F_lambda[D] = F0 + lambda J[D-D_tar].

For unrestricted spin densities, the conventional spin extension is

    E_lambda[Da,Db] = sum_s Tr(F0_s D_s)
                     + lambda sum_s Tr((D_s-D_tar_s) J[D_s-D_tar_s]),
    F_lambda_s = F0_s + 2 lambda J[D_s-D_tar_s].

The factor2 is essential; it also makes identical alpha/beta solutions agree
with the restricted spin-summed equation. A strictly increasing, nonnegative
lambda schedule provides continuation. Each stage uses the existing shared
`iterate_eager` DIIS driver. Its map acts on orthonormal-frame Fock matrices;
`occupied_projector` constructs densities, so internal occupied/virtual
rotations do not require differentiating arbitrary eigenvector columns.
Complete final orbitals use `solve_hermitian`. No separate eigensolver, DIIS
implementation, AO integral engine or copy of a PySCF SCF kernel is introduced.

The converged stage state is unshifted. Final spectral and physical-map
residuals are checked in the same orthonormal frame. `converged` diagnoses the
finite-lambda SCF equation, whereas `valid` additionally requires the requested
Coulomb density error, optional relative real-space L2 error and an external
occupied/virtual gap. A failed reference is `None`; numerical diagnostics stay.
The reported density metric is

    C = 0.5 sum_s Tr(deltaD_s J[deltaD_s])  (Hartree),

using one spin-summed component for RKS. This common diagnostic is not the
upstream printed C, whose unrestricted spin prefactor differs. Increasing
lambda usually reduces density error but does not by itself prove convergence
or noninteracting representability in the selected finite basis.

## Fixed guides and physical XC labeling

`guide_fock` constructs one fixed base, with N the total electron count:

- `xc=None`: F0=hcore, matching KSPies' no-guide convention.
- `xc='fermi_amaldi'`: F0=hcore+(1-1/N)J[n_tar]. The same total Hartree field
  enters both UKS channels.
- A semilocal named recipe or external `Functional`: F0_s=hcore+J[n_tar]
  +vxc_s[n_tar]. The potential is obtained by the existing scalar energy AD
  helper; external network parameters and density-input dependencies remain
  live. No architecture is built into iKS.

An arbitrary density-matrix energy does not establish a local multiplicative
potential; callers must declare that property before interpreting its guide
as a standard KS guide. Named hybrids and meta-GGAs are rejected by this helper rather than
silently discarding exact exchange. Existing hybrid/GKS assessment interfaces
remain separate from a validated inverse-GKS algorithm.

The ZMP correction lambda J[deltaD], or2lambda J[deltaD_s], is not the total XC
potential. If the caller supplies hcore, the returned XC operator is explicitly
F-hcore-J[n_recovered]. This finite-lambda decomposition uses the recovered
Hartree density, not an unstated replacement by the target density.

## Connection to existing APIs and remaining method boundaries

Use the existing entry point:

```python
base = iks.guide_fock(target, hcore, apply_j, xc='fermi_amaldi')
result = iks.invert(target, base_fock=base, method='zmp',
                    coulomb=apply_j, hcore=hcore, grid=grid,
                    lambdas=(1., 5., 10., 20., 50., 100.))
if result.valid:
    reference = result.reference  # existing OrbitalReference / evaluate / Trainer
```

The callable Coulomb contract admits packed ERI, native direct J/K or DF
consumers without materializing an ERI inside iKS. Its backend/workspace budget
is owned by that provider; the iKS memory preflight includes its own matrices
and DIIS history, not arbitrary callback internals. Molecular validation uses
float64 CPU. Inverse-map AD/JIT is explicitly unsupported; model energy,
potential and response AD against the recovered fixed reference remains live.

Existing `wu_yang`, `density_match` and `thermal_matrix` dispatch is retained.
The current WY AO-operator regularizer is not KSPies' gradient-PBS regularizer.
A standard gradient penalty must address potential-basis directions invisible
to the finite orbital basis, not merely rename or project the existing norm.
KSPies itself labels its hybrid `wyscf` outer loop as heuristic; copying it
would not establish a general inverse-GKS method. Those extensions are not
claimed by this first ZMP increment.

## Primary references and independent checks

- [Zhao, Morrison and Parr, Phys. Rev. A50,2138(1994)](https://doi.org/10.1103/PhysRevA.50.2138).
- [Wu and Yang, J. Chem. Phys.118,2498(2003)](https://doi.org/10.1063/1.1535422).
- [Heaton-Burgess et al., Phys. Rev. Lett.98,256401(2007)](https://doi.org/10.1103/PhysRevLett.98.256401).
- [Nam et al., KS-pies toolkit, J. Chem. Phys.154,124122(2021)](https://doi.org/10.1063/5.0040941).

`examples/iks/zmp.py` generates GradSCF PBE targets once and solves finite-lambda
RKS/UKS equations. `examples/iks/compare_kspies_zmp.py` imports only KSPies,
PySCF and NumPy, consuming the same saved targets. It independently checks
AO overlap/hcore, recovered densities and physical Fock matrices. Different
iterative trajectories are permitted; convergence and density accuracy are
reported independently. Numeric acceptance results must come from the saved
execution logs, not from an optimizer success flag alone.


Verified first increment (CPU,float64,2026-11-10):159tests pass;12independent
KSPies cases atlambda1,5,10,20,50,100 pass. Maximum AO-density discrepancy
3.71e-10, physical Fock discrepancy4.86e-10Ha. Atlambda100 the relative grid
errors are4.07e-4 (HeH+) and6.04e-4 (Li, separate spins), at fixed1e-3gates.
The external process adapts SciPy's legacy angular function name and uses
KSPies' public diis_space2/level_shift=max(0.2,lambda) controls to avoid
singular tiny-basis histories. The original solver source is not patched.
JAX scipy primitives are available, while documented jax.scipy.optimize.minimize
supports BFGS/JIT but not differentiation of the optimum; solver backward
contracts must be maintained explicitly by the shared GradSCF solver layer.
