# Inverse Kohn–Sham in GradSCF: research and implementation proposal

Date: 2026-10-25. Source baseline: `0f0fe2e298db202fe251d620f5b438655aa552cd`.
Worktree: `artifacts/side/iks`; branch: `feat/iks`.
Intended implementation: `src/gradscf/dft/iks/`.

**Status: research and proposed design, not an implemented or numerically
validated module.** No production Python, dependencies, numerical jobs or
reference-software calculations were added during this investigation.

## 1. Recommendation and scope

Start with regularized Wu–Yang inversion for real, collinear molecular systems,
including restricted and unrestricted densities. Use a fixed target density,
an explicitly local potential representation, a declared potential gauge and
occupation-subspace response. Add ZMP as an independently formulated comparison
after the primary inversion and its derivatives work. Retain a common contract
for periodic, finite-temperature and inverse-GKS extensions, without declaring
those extensions supported before separate tests.

The implementation should belong to DFT, not to a particular neural-network
model. It should export teacher quantities usable by the existing
`fixed_density`, `explicit` and `implicit` training modes. The inversion and
the subsequent training calculation are distinct differentiable problems.

This recommendation is an engineering inference from the literature and the
current repository. It is not a claim that Wu–Yang is universally more accurate
than real-space inversion or Moreau–Yosida methods.

## 2. Evidence inspected

### 2.1 Existing local wiki

The research wiki is located at
`/Users/jiaoyuan/Documents/ChatGPT/Global LLM+wiki/.wiki/`.
Its index and operation log were consulted before the following maintained pages:

- `wiki/concepts/inverse-kohn-sham-dft.md`: target-density inversion, gauge and
  energy decomposition; reviewed 2026-10-08.
- `wiki/queries/inverse-dft-and-gradscf-differentiable-training.md`: offline
  inversion versus candidate self-consistency; reviewed 2026-09-30.
- `wiki/concepts/Automatic implicit differentiation of solver roots.md`:
  residual Jacobians, tangent/adjoint equations and conditioning.
- `wiki/queries/stationarity-constrained-xc-vs-differentiable-dft-2021.md`:
  fixed inverse-KS orbitals versus differentiating a candidate SCF solution;
  reviewed 2026-10-22.

These are **secondary research records**. In particular, the September training
API descriptions are historical and were rechecked against the current source.
The wiki was read without modifying its raw artifacts or maintained pages.

### 2.2 Primary literature and reference implementations

| Reference | Direct inspection and relevance |
| --- | --- |
| Wu and Yang, 2003, [DOI 10.1063/1.1535422](https://doi.org/10.1063/1.1535422) | Original method identity verified through reference implementation and later primary accounts; original publisher full text was not accessible. |
| Zhao, Morrison and Parr, 1994, [DOI 10.1103/PhysRevA.50.2138](https://doi.org/10.1103/PhysRevA.50.2138) | Original penalty-method identity verified through KS-pies; original publisher full text was not accessible. |
| Heaton-Burgess, Bulat and Yang, [PRL 98, 256401 (2007)](https://arxiv.org/abs/cond-mat/0610884v3) | Finite orbital/potential basis imbalance and smoothness regularization; abstract and primary bibliographic record checked. |
| Jensen and Wasserman, [Numerical Methods for the Inverse Problem of Density Functional Theory](https://arxiv.org/abs/1703.04553) | Numerical/error-analysis background and finite/periodic one-dimensional benchmark leads; not a verified GradSCF implementation. |
| Shi and Wasserman, [Inverse Kohn–Sham Density Functional Theory: Progress and Challenges](https://arxiv.org/html/2103.04238v2) | Sections II–VIII: WY/PDE-CO, basis balance, guide potentials, regularization and response null spaces. |
| Nam et al., [KS-pies: Kohn–Sham Inversion Toolkit](https://arxiv.org/html/2008.08783v2) | Restricted/unrestricted WY and ZMP, user-defined potential bases and density inputs; primary algorithm and software sections inspected. |
| [KS-pies source](https://github.com/ssnam92/KSPies) | `kspies/wy.py`: potential expansion, analytic gradient/Hessian and regularization; `kspies/zmp.py`: penalty continuation and SCF assembly. Apache-2.0 repository. |
| Shi et al., [n2v, WIREs Comput. Mol. Sci. (2022), e1617](https://doi.org/10.1002/wcms.1617) | Method-suite documentation and [official source](https://github.com/wasserman-group/n2v) inspected; WY, PDE-CO, ZMP, direct, Ou–Carter and mRKS have different input requirements. BSD-3-Clause repository. |
| Kanungo, Zimmerman and Gavini, [Nature Communications 10, 4497 (2019)](https://www.nature.com/articles/s41467-019-12467-0) | State/adjoint constrained optimization, Eqs. (4)–(9); orthogonality resolves singular shifted Hamiltonians. |
| Herbst, Bakkestuen and Laestadius, [PRB 111, 205143 (2025)](https://arxiv.org/html/2409.04372v3) | Moreau–Yosida inversion for periodic systems, function-space metric and noise bounds. [Supporting repository](https://github.com/mfherbst/supporting-my-inversion) identified. |
| Bakkestuen et al., [arXiv:2606.19471v1](https://arxiv.org/html/2606.19471v1) | Periodic homogeneous-Sobolev/proximal formulation; a preprint, not a blanket guarantee for a Gaussian implementation. |
| Sheng, [arXiv:2603.23452v1](https://arxiv.org/html/2603.23452v1) | Sections II–III unify constrained search, dual optimization, penalties and state constraints; representability/gauge/weak-gap distinctions. Preprint. |
| Chai and Luber, [arXiv:2603.22140v1](https://arxiv.org/html/2603.22140v1) | Gaussian-basis Löwdin density-matrix penalty, Eqs. (6)–(9); useful matrix-domain comparison, not automatically a unique real-space multiplicative potential. Preprint. |
| Subramanian et al., [arXiv:2609.15845v1](https://arxiv.org/html/2609.15845v1) | Inverse GKS with a self-consistent nonlocal exchange operator and local remainder, distinct from local iKS. Preprint. |

Primary method sections and selected source functions were inspected, not every
page of every reference. Upstream branch URLs are mutable; they are research
locators, not numerical benchmark provenance. Exact upstream commits, input
files and outputs must be pinned before an external numerical comparison.
No upstream Python was imported or executed. Formulas below are an independent
derivation for the proposed GradSCF conventions.

An optional command-line source-archive retrieval stalled and was stopped.
Two n2v method files were retained as text with SHA-256 hashes under the ignored
`artifacts/iks-research-20261025/` evidence directory; no completed repository
clone or upstream commit-pinned benchmark is claimed. This does not change the
primary web/source passages used in the comparison above.

## 3. The object being inverted

For molecular spin channels, seek local multiplicative potentials satisfying

\[
\left[-\tfrac12\nabla^2+v_s^\sigma(\mathbf r)\right]\phi_{i\sigma}
=\epsilon_{i\sigma}\phi_{i\sigma},\qquad
n_\sigma(\mathbf r)=\sum_i f_{i\sigma}|\phi_{i\sigma}(\mathbf r)|^2.
\]

The target may be supplied as a grid density or as an AO density matrix whose
diagonal kernel defines that grid density. An AO density matrix is an input
representation, not a requirement to reproduce every element of a correlated
one-particle density matrix with a determinant.

For real AOs,

\[
n_\sigma(\mathbf r)=\sum_{\mu\nu}D^\sigma_{\mu\nu}
\chi_\mu(\mathbf r)\chi_\nu(\mathbf r),\qquad
N_\sigma=\operatorname{Tr}(S D^\sigma).
\]

Validate electron numbers, symmetry, finite values and the declared AO/grid
representation. Do not silently normalize an incorrect target. A correlated
target can have fractional natural occupations; do not impose determinant
idempotency on that input. Conversely, exact equality of its entire 1-RDM to
an integer-occupation KS 1-RDM is generally impossible.

The result includes the recovered density, kinetic energy, local potential
representation and diagnostics. When a consistent reference **total energy**
is also supplied, an approximate inversion permits the labeled decomposition

\[
E_{xc}^{\rm ref}=E_{\rm tot}^{\rm ref}-E_{nn}-T_s
-\int v_{\rm ext}n_{\rm tar}-E_H[n_{\rm tar}].
\]

The reference energy convention must say whether nuclear repulsion is included.
Do not use the KS eigenvalue sum as the physical total energy. Density inversion
alone does not supply a numerical correlated total-energy label. Report the
remaining inversion/discretization error in the derived XC label.

## 4. Wu–Yang in a finite AO and potential basis

### 4.1 Potential representation

Separate the orbital basis from the potential basis:

\[
v_s^\sigma=v_{\rm base}^\sigma+\sum_p b_{p\sigma}g_p,\qquad
(B_p)_{\mu\nu}=\int\chi_\mu g_p\chi_\nu.
\]

For each trial coefficient vector, solve

\[
F^\sigma(b)C_\sigma=SC_\sigma\epsilon_\sigma,\quad
F^\sigma=T+V_{\rm base}^\sigma+\sum_p b_{p\sigma}B_p,\quad
C_\sigma^\dagger SC_\sigma=I.
\]

For local iKS with a fixed guide, this is a **noninteracting eigenproblem** at
each potential update. It does not require a nonlinear candidate-XC SCF inside
every WY evaluation. Inverse GKS with orbital-dependent exchange does require
an inner self-consistent solve and belongs to a separately advertised variant.

A molecular Fermi–Amaldi guide can use
`v_base = v_ext + (1 - 1/N) v_H[n_target]`. Then its XC guide contribution is
`-v_H[n_target]/N`; this convention must not be confused with the full guide.
Other fixed guides are possible, including a chosen conventional potential.

### 4.2 Objective, gradient and Hessian

With the eigenstate density evaluated at the current potential, define

\[
W(b)=T_s[D(b)]+\sum_\sigma\int v_s^\sigma(b)
\left[n_\sigma(b)-n_{{\rm tar},\sigma}\right].
\]

Equivalently, subtract the target expectation of the **full local potential**
from the noninteracting eigenvalue sum. Dropping fixed target terms changes the
reported objective offset, not its coefficient gradient.

Hellmann–Feynman cancellation gives

\[
\partial_{b_{p\sigma}}W
=\int g_p(n_\sigma-n_{{\rm tar},\sigma})
=\operatorname{Tr}\left[B_p(D_\sigma-D_{{\rm tar},\sigma})\right].
\]

Choose minimization of

\[
L(b)=-W(b)+\tfrac\alpha2(b-b_0)^T K(b-b_0),
\]

where the gauge-reduced metric/regularizer is explicitly declared. A spatial
gradient metric favors smooth potentials; it is not automatically positive
definite on every finite representation.

At a gapped integer-occupation state, the one-spin response is

\[
\chi_{pq}^\sigma
=2\sum_{i\in\mathrm{occ},a\in\mathrm{vir}}
\frac{\langle i|g_p|a\rangle\langle a|g_q|i\rangle}
{\epsilon_i-\epsilon_a},\qquad
H_L=-\chi+\alpha K.
\]

For a spin-summed restricted density with occupancy two, the coefficient is
four instead of two. Use one explicit occupation convention throughout; do
not transfer restricted/unrestricted factors by analogy to another package.
AD gives the implemented gradient and HVP; this expression is an independent
tiny-system oracle. Production HVPs apply the density response and project it
back to the potential basis, without storing all `occ × vir × npotential`
products or a dense coefficient Hessian.

### 4.3 Gauge and null spaces

Adding a constant changes the AO Hamiltonian by `c*S`, leaves the fixed-N density
unchanged, and shifts all orbital energies. Remove that direction before any
inverse-Hessian solve. Under separately fixed spin populations there is a
constant gauge in each active spin channel.

Do not merely delete the first potential function and assume it is a constant.
Represent a constant analytically as `c*S`; measure numerical grid-overlap
error separately. A finite Gaussian potential basis need not contain an exact
constant, but can still contain nearly constant directions.

Finite orbital spaces create additional invisible/weakly visible potential
directions. Gauge fixing alone does not repair these. Diagnose the observable
response rank and regularization metric. Small reference problems can use SVD;
large problems need bounded operator-based condition estimates. A truncated
rank is fixed metadata for a derivative calculation, with an explicit validity
boundary at rank changes. An unregularized singular response must fail clearly;
a hidden pseudoinverse or denominator broadening is not an exact iKS derivative.

With nonzero regularization the optimum generally has a nonzero density-moment
error. Report the regularized stationarity residual and the unregularized
density error separately. Neither is a substitute for held-out-grid potential
quality, guide/PBS sensitivity or the occupied–virtual spectral gap.

## 5. Alternatives and their roles

### ZMP

For a declared base potential, a useful convention is

\[
E_\lambda[D]=T_s[D]+\int v_{\rm base}n[D]
+\frac\lambda2\iint\frac{\delta n(\mathbf r)\delta n(\mathbf r')}
{|\mathbf r-\mathbf r'|}\,d\mathbf r\,d\mathbf r',\quad
\delta n=n[D]-n_{\rm tar}.
\]

Its added potential is `lambda*v_H[delta_n]`. This supplies a route through
existing Coulomb-only direct J/K or DF actions without a saved full ERI.
Increase penalty strength using continuation; retain every finite-strength
density mismatch and conditioning diagnostic. A finite-penalty solution and an
extrapolated infinite-penalty result are different derivative objects.
Reconstruct `v_xc = v_s - v_ext - v_H[n_target]` using the exact guide convention.

Use ZMP for an independent method comparison, not as a disguised WY objective.
Reuse shared nonlinear/DIIS and implicit-root machinery instead of copying
KS-pies' method-local DIIS implementation.

### PDE-constrained density optimization

Minimize a density misfit plus potential regularization, subject to KS state
equations and orbital orthogonality. State/adjoint equations provide gradients.
This is a valuable later alternative for weighted real-space matching.

The inspected n2v `pdeco.py` requests a four-index overlap object `S4` and solves
shifted Hamiltonian equations for individual orbitals. GradSCF should instead
integrate density errors blockwise and use projected/bordered response solves.
A raw solve of `F-epsilon*S`, followed by projection, is not a general treatment
of the singularity, especially for degenerate occupied states. The original
state/adjoint equations are the theoretical reference, not a storage layout to
copy. No PDE-CO accuracy equivalence is claimed for a Gaussian WY implementation.

### Moreau–Yosida and periodic inversion

The 2025 periodic work and 2026 follow-up distinguish density/potential dual
spaces and regularized proximal densities. They provide a strong second-stage
reference for reciprocal-space inversion. Their error bounds depend on their
specific mathematical setting and cannot be attached to an arbitrary ridge fit.

Periodic adaptation must preserve the forward calculation's pseudopotential,
nonlocal ionic terms, k weights, cell normalization, electrostatic kernel and
G=0 convention. Metals require a declared temperature/ensemble and a chemical
potential response. A molecular Fermi–Amaldi tail is inappropriate for bulk.
Low-dimensional cells require their actual Coulomb boundary conditions.

### Density-matrix penalties and inverse GKS

The 2026 matrix-penalty proposal motivates basis-covariant mismatch metrics,
such as comparing `S^(1/2) D S^(1/2)`. However, a free Hermitian potential matrix
is not automatically a uniquely defined real-space local potential. Full 1-RDM
matching also differs from matching its diagonal density, particularly for
correlated non-idempotent targets. Treat this as an explicit matrix-domain
variant or diagnostic, not a replacement definition of the local iKS API.

For inverse GKS, define the nonlocal operator and its orbital dependence first.
The 2026 inverse-GKS paper self-consistently updates exchange inside the outer
local-potential inversion. Freezing inverse-KS orbitals for later hybrid fitting
does not implement that procedure. mRKS/other wavefunction inversions also need
more than a density alone and are separate future input contracts.

## 6. Differentiating the inversion

Let `theta` contain differentiable target-density data, reference integrals,
potential-basis parameters, geometry and any declared regularization parameters.
Use gauge-reduced coordinates `b=Z*z` and define

\[
R(z,\theta)=Z^T\nabla_b L(Zz,\theta)=0,\qquad
A=\partial_z R.
\]

For a perturbation `dot(theta)` and outer scalar observable `ell`,

\[
A\dot z=-R_\theta\dot\theta,\qquad
A^T\lambda=\ell_z,\qquad
\frac{d\ell}{d\theta}=\ell_\theta-\lambda^T R_\theta.
\]

Attach this residual with the existing shared `attach_root`; use checked shared
linear solves for JVP/VJP and nested response. Do not differentiate optimizer
history for the implicit contract. An explicit finite-iteration derivative is
different and must be labeled separately if offered.

For geometry response, keep moving S/T/external integrals, potential-basis
evaluation, quadrature coordinates/weights and the chosen target-density path
live. A fixed AO density matrix while moving the AO basis is a particular
target path, not automatically the derivative of a recomputed FCI density.

Within occupied or virtual degeneracies, differentiate the occupied projector
or its block response rather than individual canonical eigenvectors. The
existing `dfpt.ks.ks_problem` demonstrates a suitable fixed-occupation chart.
For WY, set the Hamiltonian callback independent of the trial density: its
response is the **unscreened noninteracting chi_s**, not the screened response
of an interacting candidate DFT functional.

These derivatives need a converged stationary point, fixed response rank,
controlled regularization and a separated occupied subspace. A closing Fermi
gap or changing occupation invalidates the zero-temperature smooth branch.
Finite-temperature response adds occupation and fixed-N chemical-potential
derivatives. Existing `fermi_density` and `matrix_function` document first-order
degeneracy support; they do not establish higher-order response at degeneracy.

Keep three objects distinct:

- `chi_s = delta n / delta v_s`: response used inside local inversion;
- the gauge/regularization-dependent derivative of the inverse map;
- `f_xc = delta v_xc / delta n`: a model's XC functional kernel.

They must not be interchanged in TDDFT, DFPT or training.

## 7. Current GradSCF infrastructure and concrete gaps

| Component inspected at the baseline | Reuse or required work |
| --- | --- |
| `dft/__init__.py` | Lazy facade; add the `iks` submodule when implementation starts. |
| `integrals/grids/blocks.py: AOGrid.block/reduce` | Differentiable AO tiles and checkpointed consumers; reuse for density/moment/potential contractions. |
| `integrals/backends/native/jk.py` | Existing direct J/K and geometry products; Coulomb-only guide/ZMP actions should use this API. |
| `dfpt/ks.py: ks_problem` | Fixed-occupation Hermitian projector chart; useful response building block. The eager DFPT facade is more restricted than the low-level builder. |
| `solvers/nonlinear/root.py: attach_root` | Shared converged-root JVP/VJP; no iKS-specific linear solver. |
| `solvers/nonlinear/iterate.py` and shared linear solvers | Existing residual/DIIS/budget mechanisms; reuse where appropriate. |
| `solvers/nonlinear/minimize.py` | Current optimizer specializes in orbital rotations. A general potential-coefficient minimizer is not already provided by that interface. |
| `solvers/eigen/response.py` | Subspace response mechanisms and validity checks; no need for an isolated-orbital derivative inside a degenerate occupied cluster. |
| `pbc/occupations.py` | Canonical finite-T chemical-potential response and Fermi density, with documented derivative limits. |
| `dft/functional.py` and `dft/derivatives.py` | External scalar-energy callbacks, physical AO potentials and XC Hessian actions. |
| `model/neural_xc/training/{config,trainer,targets}.py` | Existing three training modes and `Sample`; reuse instead of introducing an iKS trainer or another model package. |

The potential-coefficient optimization gap must be solved in the **shared
solver layer** if the current residual-driven iteration is insufficient.
Generalize only the necessary bounded scalar optimization/line-search pieces;
do not force coefficients through an orbital-rotation API or duplicate SciPy
optimizer/DIIS implementations under iKS. SciPy can remain an external oracle.

Two inspected training behaviors require targeted compatibility work:

1. `xc_potential_matching_penalty` accepts a two-spin target but averages its
   channels. That does not preserve unrestricted iKS information.
2. This penalty sanitizes nonfinite potentials with `nan_to_num`; inversion
   failures must instead remain invalid teacher data and be rejected before
   training. A failed inversion is not a finite zero-potential label.

The existing periodic `Sample` adapter also rejects potential/kernel targets;
periodic teacher-potential training needs its own explicit extension.
Unrestricted inversion output likewise requires an audited spin-preserving
sample/loss adapter; an unrestricted forward inversion does not by itself prove
unrestricted support in every existing Trainer target.

An arbitrary external neural energy of a full density matrix can produce a
general AO operator. This need not be the projection of a multiplicative local
potential. Compare a supported model's physical AO operator, or its occupied–
virtual stationarity residual, with matching conventions. A pointwise local
potential loss is appropriate only when that model defines the required local
Euler derivative. Hybrid and kinetic-density models need an explicitly labeled
GKS decomposition.

## 8. Proposed package and API

```text
src/gradscf/dft/iks/
    __init__.py     # small public exports
    api.py          # eager molecular facade and pure array entry
    types.py        # target/problem/config/result PyTrees
    potential.py    # local potential map, transpose, gauge and metric
    ks.py           # noninteracting state and shared projector response bridge
    wy.py           # objective, residual and method adapter
    response.py     # attach the inversion root using shared solvers
    diagnostics.py  # density, rank, gap and potential-quality diagnostics
    zmp.py          # added only in the comparison stage
```

This is a proposed layout, not created Python scaffolding. No `solver.py` or
method-local eigensolver/GMRES/DIIS is planned. Do not split every contraction
into a module. Keep a single potential-map contract shared by forward, AD and
diagnostics; numerical routines belong in src, not examples/reproducibility.

Proposed facade spelling, subject to design review:

```python
from gradscf.dft import iks

# dm_target is in the stated AO basis, not an unconverted CI MO density.
inverse = iks.IKS(mol, density=dm_target, energy=reference_total_energy,
                  method="wy", max_memory=60000.).run()
```

Also provide one pure array entry, `iks.invert(problem, config=...)`, so geometry
and target-data differentiation do not depend on an eager object's cache. Do
not expose several long synonymous constructors. Potential basis, gauge,
regularization and occupation policy must be inspectable configuration, even
when a facade supplies sensible defaults.

Results should distinguish: numerical convergence; unregularized density error;
regularized stationarity; response validity/rank/gap; chosen potential gauge;
local vs GKS/operator representation; reference energy and its decomposition;
potential-basis/grid provenance. Return invalidity explicitly for unresolved
roots; successful optimizer termination alone is insufficient.

## 9. Memory and training contracts

Potential application and transpose projection should operate in grid/basis
tiles. Retain necessary AO matrices and requested result arrays, not the full
`npotential × nao × nao` tensor, `grid × occ × vir` array or an `nao^4` density
misfit tensor. A tiny dense reference may store these for validation only.

Inherit the existing default `max_memory=60000` decimal MB and charge the
orbital state, potential data, optimizer/Krylov history and supported AD state
before selecting tiles. Keep numerical quadrature, basis and occupations
fixed when changing the memory budget. Explicit array ownership remains a
planning bound, not an OS RSS/VRAM cap. Use the existing explicit device contract.

An iKS-to-training adapter should return existing sample data plus typed label
metadata, not construct a second trainer. Preserve total-energy supervision;
any XC decomposition/potential supervision is an additional declared target.
Keep spin channels and potential gauges explicit.

Offline teacher generation can stop gradients deliberately. Differentiable
teacher generation must retain the full iKS residual response. Subsequent
`explicit` and `implicit` modes continue to solve/differentiate the **candidate**
SCF using their existing semantics. Fitting energy or stationarity on frozen
inverse orbitals does not certify candidate self-consistency or global stability.

## 10. Finite implementation and TDD sequence

These are proposed tasks, not acceptance marks. Retain one child level and at
most four children per parent; record exact tests and evidence when executed.

- [ ] **IKS-01: Target, local-potential and gauge contract.**
  - [ ] Validate AO/grid targets, spin counts, units and optional energy labels.
  - [ ] Check potential apply/transpose duality and constant `c*S` invariance.
  - [ ] Add dense/tiled parity, tail blocks and insufficient-budget preflights.
- [ ] **IKS-02: Restricted/unrestricted WY forward.**
  - [ ] Add the smallest necessary shared coefficient optimizer support.
  - [ ] Compare AD gradient/HVP with analytic spin-factor and finite-difference oracles.
  - [ ] Recover manufactured local potentials/densities with regularization/rank diagnostics.
  - [ ] Cross-check KS-pies on pinned identical orbital/PBS/guide conventions.
- [ ] **IKS-03: Implicit response and geometry.**
  - [ ] Attach the gauge-reduced root and check target-data JVP/VJP duality.
  - [ ] Check nested HVP and two FD steps on smooth fixed-rank examples.
  - [ ] Check occupied/virtual degeneracy invariance and genuine gap-closing rejection.
  - [ ] Check moving AO/grid/potential data and CPU/GPU parity on explicit devices.
- [ ] **IKS-04: Independent method and quality assessment.**
  - [ ] Add finite-penalty ZMP continuation using existing Coulomb/SCF paths.
  - [ ] Compare density, kinetic energy and gauge-aligned observables on common targets.
  - [ ] Scan grid/PBS/regularization/noise sensitivity; validate on a separate grid.
- [ ] **IKS-05: Existing functional-training integration.**
  - [ ] Preserve spin-resolved teacher labels and reject invalid inverse results.
  - [ ] Compare supported AO/grid potential or stationarity losses with explicit gauges.
  - [ ] Validate unchanged fixed-density/explicit/implicit candidate-training modes.
- [ ] **IKS-06: Examples, measurements and final capability audit.**
  - [ ] Add thin H2/He and H2O examples, with an open-shell Li/OH case; no CLI.
  - [ ] Include FCI-derived density/total-energy data and clearly label representability limits.
  - [ ] Measure compile/warm/RSS/VRAM on a small increasing-size family.
  - [ ] Document periodic/finite-T/inverse-GKS boundaries and retained numerical failures.

Begin with deterministic manufactured potentials to isolate numerical correctness.
Then use semilocal forward-density round trips, followed by correlated targets.
Same-discretization round trips are controlled verification, not evidence that
arbitrary correlated densities are representable or that potential tails are
accurate. For initial independent FD tests, target coefficient gradients/HVPs
near1e-7/1e-6 absolute agreement in declared units, using two step sizes and a
well-conditioned fixture; refine tolerances from conditioning rather than
loosening them to hide failures. Establish density and potential thresholds
separately from grid convergence and external-oracle comparisons.

All future numerical tests remain remote in the user-authorized Slurm pool and
resource limits. This research stage performed source/document inspection only.
The existing worktree baseline is clean, but no fresh numerical baseline run,
new iKS result, method benchmark or implemented feature is claimed here.

## 11. Decisions needed before production coding

Recommended first acceptance slice: real molecular R/U, fixed integer
occupations with an occupied–virtual gap, a local grid-evaluable PBS,
regularized WY and an independently testable implicit backward. Keep periodic,
finite-temperature, inverse GKS and full wavefunction-driven alternatives as
separate advertised stages rather than silent fallbacks.

The immediate design review should settle the default PBS/guide and explicit
regularization policy, together with the initial scope. Their implementation
can then follow the finite TDD sequence above. There is no basis-independent
universal regularization constant justified by this literature survey.
