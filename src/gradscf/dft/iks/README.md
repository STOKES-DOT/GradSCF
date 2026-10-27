# Inverse references and functional assessment

The package provides physical targets, block-local potential maps, SCF metadata
snapshots, an explicitly selected finite-temperature matrix-reference generator
and fixed-reference functional evaluation. Zero-temperature Wu–Yang and coupled
inverse-GKS optimization/backward remain unfinished in
[the research plan](../IKS_RESEARCH.md). The matrix-reference method does not
silently substitute for either of those methods.

## AO target conventions

```python
from gradscf.dft import iks

target = iks.Target(dm_ao, overlap, nelec=2, layout="rks", energy=reference_energy)
print(target.electron_numbers)
print(target.valid)
```

All quantities use atomic units. `energy`, when supplied, is the reference
total energy including nuclear repulsion. A CI density supplied in an MO basis
must first be transformed into the stated AO basis.

| Layout | Density shape | Electron-count argument | Component order |
| --- | --- | --- | --- |
| `rks` | `(nao, nao)`, real, spin summed | Even integer `N` | Total density |
| `uks` | `(2, nao, nao)`, real | `(Nalpha, Nbeta)` | Alpha, beta |
| `gks` | `(2*nao, 2*nao)`, Hermitian, possibly complex | Integer total `N` | Charge, mx, my, mz |

The spinor AO order is alpha then beta. The overlap is the spatial overlap.
Concrete invalid inputs raise; inputs reconstructed inside JIT retain an
explicit `valid` flag. Checks include Hermiticity, positive overlap, electron
counts, finite optional energy and metric-weighted natural occupations. They
do not impose determinant idempotency on a correlated target or silently
renormalize the density. Floating-point physical checks use a floor of1e-9
and100 times the density dtype's machine epsilon.

`Target.components` retains the original density dependence; the record is a
PyTree. A valid target is not a proof of noninteracting local v-representability.
Grid-only targets and fractional-electron ensembles are not provided yet.

## Local potential and its transpose

```python
potential = iks.Potential(
    grid_ao, overlap, basis_functions, nparam,
    params=basis_parameters, layout="gks", max_memory=60000.,
)
matrix = potential.matrix(coefficients)
moments = potential.moments(dm_spinor)
```

`basis_functions(params, coordinates)` is a pure, block-local callback returning
real values with shape `(block_points, nparam)`. All differentiable evaluator
data belong in `params`; closures are for static metadata. The AO supplier is
the existing `AOGrid`. Matrix construction and moments use its checkpointed
block reduction and identical quadrature. No full potential-basis/AO-pair
tensor is stored.

RKS coefficients have shape `(nparam,)`, UKS `(2, nparam)`, and GKS
`(4, nparam)`. Spinor fields are assembled as

\[
U=\begin{pmatrix}V_0+V_z&V_x-iV_y\\V_x+iV_y&V_0-V_z\end{pmatrix}.
\]

The physical transpose obeys

\[
\sum_{cp}b_{cp}\,\mathrm{moments}(D)_{cp}
=\operatorname{Re}\operatorname{Tr}[U(b)D].
\]

`constant_potential(overlap, value, layout=...)` adds the analytic `c*S`
constant independently of quadrature-overlap error. `matrix(..., constant=...)`
uses the same helper. A GKS constant is scalar; a uniform magnetic field is not
silently removed. UKS accepts two constants for its separately fixed spin
populations. This is a constant-channel convention, not an automatic treatment
of the potential basis's additional response null space.

`max_memory` uses the common decimal-MB policy and runtime reserve. Known
provider, parameter, input, accumulator and assembly arrays are charged before
choosing a block, with precision following the grid and evaluator operands.
The known input/AO minimum is rejected before tracing the basis callback;
the evaluator's output precision is then checked before selecting the final block.
Changing the budget changes block width, not the grid, basis or coefficients.
This first map budget is not a complete future inversion/AD-state or process
RSS/VRAM limit. Arbitrary callback activations and compiler allocations remain
outside its known-array estimate.

## SCF and custom-XC metadata

```python
reference = iks.Reference.from_source(mf)  # RKS, UKS or GKS
target = reference.target(dm_target_ao, energy=reference_energy)
```

Taking the snapshot does not call `kernel`, `to_reference`, or prepare complete
AO/MO ERIs. It retains the supplied `TraditionalXCFunctional` or external
`Functional` object and explicit `params`; strings go through the existing
`custom_xc` factory. Complete full-range/SR/LR exchange recipe metadata and
additional physical terms are preserved. The budget inherits from `mf` unless
overridden. Actual spinor overlap blocks are reduced to their shared spatial
overlap after validating the block structure; a spinor core operator is retained.

The source must be converged. Existing SCF signatures are checked for changed
source settings; GKS now records its signature after facade execution, and the
snapshot separately requires convergence. Reference objects are eager
preparation snapshots, not an implicit SCF primitive.

Preserving an RSH recipe or neural callback does not execute it. In particular,
the existing GKS forward restrictions on RSH and external `Functional`, the
shared complex custom-functional adapter, hybrid/custom composition, and the
coupled inverse-GKS backward remain explicit implementation work. The package
does not introduce another XC registry, trainer or numerical solver.


## Explicit auxiliary matrix reference

```python
reference = iks.invert(
    target, potential, base_fock,
    method="thermal_matrix", temperature=.01,
    regularization=1e-12, rank_tolerance=1e-12, tolerance=1e-6,
)
```

This entry requires a full AO 1-RDM strictly inside its occupation bounds,
not just a sampled density. Temperature is in Hartree (kB=1). With normalized
orthogonal-frame occupations P, it constructs H-mu=-T*log(P/(1-P)), and projects
the resulting operator into the supplied local potential basis modulo the
constant gauges. `base_fock` is a prescribed fixed operator; it is not updated
as a hybrid/custom inner SCF. Target occupations are never clipped into range.
Integer/idempotent boundary targets must use a different formulation.

`Potential.ao_products(grid, overlap, layout=...)` supplies a small local PBS
from products of real Gaussian AOs. Products are evaluated blockwise. For this
explicit small-system method, the coefficient-space Gram matrix is dense and
bounded by the shared linear config's max_dense (default256 coefficients).
That bound is checked before forming its identity/matrix; this is not a large-
system matrix-free inversion implementation.

Projection columns are normalized by their metric norm before shared spectral
factorization. Eigenvalues below rank_tolerance times the largest eigenvalue
are explicitly excluded. Regularization penalizes normalized coefficients;
projection_rank and projection_condition expose this choice. The method uses
the shared Hermitian and checked linear solvers, not a method-local solver.
Constant/null-space ambiguity and potential smoothness remain distinct issues.

The fitted operator independently generates a density through fixed-electron-
number Fermi filling, including chemical-potential determination. `valid` needs
both metric-density and gauge-aligned operator errors below tolerance, plus
converged shared solves. A small projection-equation residual alone is not a
successful inverse. Reported density_error is the maximum absolute entry of
S^(1/2)*(D_recovered-D_target)*S^(1/2), and operator_error is the corresponding
aligned orthogonal-frame operator maximum in Hartree.

This is an **eager offline reference generator**. Traced inputs, JIT or AD
through this inverse entry raise explicitly. The returned fixed reference can
be used inside differentiable model losses. A finite-temperature fit to a
zero-temperature FCI density is an auxiliary numerical construction, not an
exact zero-temperature KS/XC potential or a thermal many-body reference.
Potential recovery is only within the declared finite basis and gauge.
Matching the full correlated1-RDM fixes its kinetic expectation as well; this
does not establish the zero-temperature density-constrained KS kinetic energy.
The evaluator therefore supervises total energy, not a mislabeled exact
zero-temperature XC-energy decomposition.

## One interface for analytic and neural functionals

```python
energy = iks.energy_function(state, functional, layout="uks", nuclear_repulsion=enuc)
score = iks.evaluate(reference, energy, params)
loss = score.energy_error**2 + abs(score.energy_error) + .01*score.operator_mse
```

`energy_function` binds an existing traditional recipe, `custom_xc` or supported
neural model to a TOTAL-energy callback, adding one-electron, Coulomb and
nuclear terms once. It uses existing SCF assembly for conventional exchange
recipes. Missing nuclear-energy metadata requires an explicit value, including
zero for an isolated atom. An object implementing energy_from_molecule returns
XC energy under this adapter; it needs no neural-specific density-update method.
Alternatively pass any pure `energy_fn(params, density) -> total energy`
directly to `evaluate`.

The evaluator uses AD to obtain the physical Hermitian AO derivative, with the
complex-gradient conjugation convention. It returns separate total-energy
error, gauge-aligned operator MSE and commutator stationarity MSE. Parameters
remain live through the operator loss, so its gradient includes mixed parameter-
density derivatives. A nonfinite model result stays invalid, rather than being
changed to a finite score. No candidate SCF is run. The commutator is a necessary
orbital-stationarity diagnostic, not proof of correct occupations, stability or
self-consistent density. There is no candidate-density error fabricated from
an unchanged reference density.

This does not introduce a fourth mode in the existing Trainer. The carbon
example uses a fixed-reference loss with standard Optax Adam; subsequent
self-consistent fine-tuning still belongs to the existing explicit/implicit
paths. Complete custom-spinor, frozen HFX/PT2 neural channels and inverse-GKS
inner-state contracts remain separately guarded.

See [the carbon example](../../../../examples/iks/carbon_fci.py) and its
[independent FCI comparison](../../../../examples/iks/compare_carbon_fci_pyscf.py).
Both are ordinary scripts with no CLI and reuse src implementations.
