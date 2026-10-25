# Inverse Kohn–Sham infrastructure

This package currently provides physical target records, local potential maps
and eager SCF snapshots. It does **not** yet solve the inverse problem. There
is no `IKS.run` or `invert` entry until the forward and implicit-response gates
in [the research plan](../IKS_RESEARCH.md) are implemented and verified.

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
