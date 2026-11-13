# Inverse references and functional assessment

## Supplied orbital references and shared fixed-density objectives

`OrbitalReference(target, coefficients, occupations, energies, method=...)`
accepts complete RKS, UKS or complex spinor orbitals. The input target may be a
correlated AO 1-RDM; its real-space density is the inversion target, and it is
not required to equal the reference orbitals' 1-RDM. Structural `valid` checks
finite data, C^dagger S C=I and electron counts, not ground-state stability,
local v-representability or a successful density inversion. `method` must
identify where the supplied orbitals came from. Store and independently verify
the source inversion's density error before admitting a production teacher.
Supplying orbitals does not execute Wu-Yang or coupled inverse-GKS.

```python
score = iks.evaluate(reference, energy_fn, params, quantities=('energy',))
value, score = iks.loss(reference, energy_fn, params,
    weights={'energy': {'mse': 1., 'mae': 1.},
             'stationarity': {'norm2': .1}})
```

`energy_fn(params, density)` returns TOTAL energy in Ha. Energy-only evaluation
does not construct dE/dD. Operator/stationarity/density requests share one live
Fock derivative. With an `OrbitalReference`, omitted quantities means energy
only; omitted quantities on legacy thermal results preserves their old
energy/operator/commutator behavior.

Available objectives have deliberately different names and units:

| Objective | Metric | Meaning |
| --- | --- | --- |
| energy | mse, mae | Total-energy error squared / absolute, Ha^2 / Ha |
| operator | mse | Orthogonal-frame total Fock error after scalar gauge removal, Ha^2 |
| xc_potential | mse | Same metric for an explicitly supplied AO XC teacher, Ha^2 |
| stationarity | norm2 | Sum over occupied/virtual rotations abs(kappa_ai)^2, dimensionless |
| predicted_density | norm2 | Integral of component density error squared, bohr^-3 |

For `xc_potential`, supply both `reference.xc_potential` and an explicit
`xc_energy_fn`; a total Fock must not be compared to an XC-only teacher.
`energy_function(..., component='xc')` exposes supported scalar custom/neural
XC and traditional R/U XC components. Its default is total energy, with nuclear
repulsion required explicitly when absent from state metadata. Spinor custom
models, separate spinor XC components, and ambiguous dispersion decompositions
remain guarded rather than inferred from a scalar callback.

For an analytic candidate that keeps a fixed exact-exchange fraction while
fitting a parameterized custom semilocal component, use
`hybrid_xc_energy_function(...)`. It is deliberately a fixed-reference-only
callback: it returns XC energy in Hartree, contracts the existing AO/DF/direct
JK representation, keeps custom parameters differentiable, and rejects GKS.
It does not add a hybrid SCF binding, a coupled inverse-GKS response, or a
Wu--Yang solver.

For wavefunction-derived XC labels such as XC100 `Exc_TZ`, use
`xc_energy_loss(..., target_xc_energy=...)`; do not place an XC-only label in
`Target.energy`, whose contract remains total energy including nuclear
repulsion.

The integer occupied/virtual model uses kappa_ai=-F_ai/max(eps_a-eps_i,gap_floor).
Default gap_floor=.05 Ha is an explicit uncoupled-response regularization,
not an exact SCF Hessian or hidden broadening in an inverse derivative. RKS
occupation factors and complex spinor density rotations are retained.
Fractional reference occupations are valid for energy/operator evaluation;
stationarity and predicted density reject that chart. Only a requested
predicted-density metric constructs delta D.

For predicted density, pass `grid=AOGrid(...)` or `(real_ao, weights)`.
Density-norm quadrature requires real finite nonnegative weights with positive
total weight; invalid quadratures cannot produce a valid loss.
The metric compares n_s+delta n with n_target and reports the separate
reference_density_norm2. It preserves alpha/beta or charge/magnetization
components, including the original inversion residual. This is an approximate
first-order diagnostic, not actual candidate SCF density error.

`loss(..., scales={...})` divides each residual by an explicit scale before
squaring/absolute value. Scales have units Ha for energy/operator/XC potential,
dimensionless for rotation and bohr^-3/2 for density norm; defaults are 1 in
these units. The returned raw Evaluation stays unnormalized. Invalid reference, nonfinite explicit parameters
or model values produce invalid scores/nonfinite loss, never zero labels or a
finite reward. The caller defines any reward transformation or RL policy.

`potential_error` separately supports real, component-by-grid local potential
data: `grid_l2` is the weighted grid point mean summed over components;
`density_integral` squares each density-weighted integral before summing (the
NNLap metric). These are different objectives. The latter requires a fixed
common gauge; removing its weighted density mean would erase its signal.
This metric helper does not invent a pointwise potential for nonlocal models.

The old `Evaluation.stationarity_mse` is retained as a commutator diagnostic
(`commutator_mse` alias). It is distinct from the new `rotation_norm2`; an AO
commutator is not used as a substitute for the occupied/virtual objective.

## Existing Trainer and agent integration

```python
sample = training.Sample(state, inverse=reference, nuclear_repulsion=enuc)
trainer = training.Trainer(functional, params=params)
trainer.mode = 'fixed_density'
trainer.loss = {'energy': {'mse': 1., 'mae': 1.},
                'stationarity': {'norm2': .1}}
trainer.run(sample, steps=20)
```

The adapter copies the reference orbitals/density into the existing prepared
state without SCF, checks the overlap and uses the existing Sample PyTree,
Adam, parameter-gradient checks, atomic update rejection and history. An energy
override on Sample is explicit. Source AO ordering must match the reference;
overlap equality alone cannot prove basis identity. All three ordinary Trainer
modes remain; inverse samples currently require fixed_density. Use ordinary
samples with explicit/implicit for actual candidate SCF validation. Mixing
ordinary and inverse samples in one dataset, additional training objectives,
and per-electron/per-atom normalization are rejected in this initial adapter.
For an inverse sample, xc_potential supervises its AO teacher, whereas ordinary
samples retain their separate grid-potential target. No fourth training mode,
optimizer or reward policy is introduced.

When an inverse XC/predicted-density objective is active, a simultaneously
supplied Sample grid-potential/density label is rejected as ambiguous. Put the
inverse AO XC teacher and target density on OrbitalReference. Inactive labels
do not affect energy-only objectives. Unperformed DFPT response diagnostics are
None/NaN, not a failed solve or a measured zero error.

Standalone `iks.loss` supports explicit per-objective scales/gap_floor. The
initial Trainer adapter uses the documented unit scales and .05 Ha gap floor;
its weights must be chosen accordingly. Missing actual SCF density diagnostics
are NaN; unrequested inverse diagnostics are NaN in history and None in the
raw Evaluation. scf_cycles=0 and scf_converged=None indicate no candidate SCF.

See [the two-AO analytic/MLP example](../../../../examples/iks/fixed_reference.py).
It is a manufactured reference/interface check, not a molecular inversion or
accuracy benchmark. The same raw metrics can feed an external agent's reward.

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

## Offline Wu-Yang local inversion

`iks.invert(..., method="wu_yang")` now implements real RKS/collinear-UKS
zero-temperature forward reference generation. See [WU_YANG.md](WU_YANG.md)
for the objective, gauge/rank and acceptance contracts, finite-basis limits,
and the verified known-PBE molecular round trip. Earlier statements below or
in historical experiment notes that all Wu-Yang work is unimplemented are
superseded for this bounded forward path. Spinor/coupled inverse-GKS and the
inverse-map backward remain unimplemented. Invalid inversions return no usable
OrbitalReference. Existing fixed-reference analytic/neural assessment and all
three ordinary training modes retain their contracts.

### Finite-penalty ZMP and fixed local guides

`invert(target, base_fock=base, method='zmp', coulomb=apply_j, ...)` accepts a
spatial Coulomb action, with no fitted potential basis. RKS and collinear UKS
use the conventional lambda and2lambda factors respectively. Lambda stages,
SCF residual, Coulomb density error and optional relative grid-density error
are separate diagnostics. Only a valid inverse exports an `OrbitalReference`.
`guide_fock` supports no guide, Fermi-Amaldi, and local classic/external scalar
XC guides using shared AD. Details, units and the explicit inverse-GKS/WY
regularization boundary are documented in [KSPies.md](KSPies.md).


## Single-sample reward without candidate SCF

Prepare a density-qualified `OrbitalReference` once, for example with
`invert(..., method='wu_yang')`. The reference stores a complete metric-
orthonormal orbital frame, integer occupations and orbital energies. A
correlated target AO 1-RDM is allowed; it need not equal the KS 1-RDM. Reference
validation checks representation and Aufbau filling, but does not certify a
local inverse or enforce a universal density-error tolerance. Admit teachers
using the source inversion's independently recorded density error.

```python
energy = iks.energy_function(state, candidate_functional, layout='rks',
                             nuclear_repulsion=nuclear_repulsion)
score = iks.reward(reference, energy, params, grid=grid,
    energy_scale=.01, density_scale=.01, weights={'gap': 1., 'density': 1.})
print(score.value, score.valid, score.density_error)
```

`candidate_functional` uses the same traditional/custom/neural energy contract
as DFT. A direct `energy_fn(params, density)` must return **total** energy in
Hartree, including kinetic, external, Hartree and XC terms. Passing XC energy
alone would diagonalize an XC-only operator and is physically incorrect. No
reference energy label is needed for this density-based reward. The optional
`score.total_energy` is evaluated at the frozen reference, not at the one-shot
or converged candidate density.

### Why a Wu-Yang evaluation needs a one-electron solve

For a fixed local base operator `h0`, write `F(v)=h0+v`. Define

\[
 E_s(F)=\min_{D\in\mathcal D_N}\operatorname{Tr}(FD),\qquad
 W_t[v]=E_s(h_0+v)-\operatorname{Tr}(vD_t).
\]

Here the minimization fills the lowest orbitals at fixed integer electron/spin
counts; it is a linear eigenproblem, not an SCF iteration. The potential is
held fixed during this solve. Its first variation is

\[
 \delta W_t=\operatorname{Tr}[(D_v-D_t)\delta v]
            =\int [n_v(\mathbf r)-n_t(\mathbf r)]\delta v(\mathbf r)d\mathbf r.
\]

Thus maximizing the concave potential functional recovers the target density
within the declared potential/orbital spaces when that density is representable.
This is the offline inversion. Each subsequent candidate evaluation performs
only:

1. Compute `E_theta[D0]` and the physical Hermitian derivative
   `F_theta = dE_theta/dD | D0` through the existing functional AD module.
2. Transform with the complete reference orbitals: `H=C0^dagger F_theta C0`.
   Since `C0^dagger S C0=I`, no new overlap diagonalization is needed.
3. Fill the lowest orbitals of this frozen operator once to obtain `D1`.
4. Reduce the density errors on the original grid, using bounded grid blocks.
   There is no evaluation of `F_theta[D1]` and no candidate SCF loop.

The implementation reports two different energy-like quantities:

\[
 G_{\rm var}=\operatorname{Tr}(F_\theta D_0)-E_s(F_\theta)\geq0,
\]

\[
 \Delta W_t=\operatorname{Tr}(F_0D_0)-E_s(F_\theta)
             -\operatorname{Tr}[(F_0-F_\theta)D_t].
\]

`variational_gap` is the first expression. It is nonnegative because `D0` is a
feasible determinant with the same electron/spin counts as `D1`. It detects
incorrect Aufbau occupation even when every occupied-virtual Fock residual
vanishes. `wu_yang_gap` is the **signed** second expression. For a common local
base and an exact density inverse, the expressions agree: their difference is
`Tr[(F0-Ftheta)(D0-Dt)]`, which vanishes for matching real-space densities.
Approximate inverses, incomplete potential spaces or nonlocal operators can
break that equality. The signed quantity is retained without clipping and is
not automatically a nonnegative error or an XC energy.

The relative density errors are

\[
 e_1^2=\frac{\int \sum_c|n_c[D_1]-n_c[D_t]|^2}
                  {\int \sum_c|n_c[D_t]|^2},\qquad
 e_0^2=\frac{\int \sum_c|n_c[D_0]-n_c[D_t]|^2}
                  {\int \sum_c|n_c[D_t]|^2}.
\]

The quadrature and units are atomic. Components are spin-summed RKS density,
separate alpha/beta UKS densities, or GKS charge and three magnetization
components. Both errors and their common normalization use one grid pass.
A zero target grid norm is invalid. `density_error=e1` and
`reference_density_error=e0`; the reference baseline is **not subtracted**.

### Reward convention, invariance and AD

\[
 L=w_G\frac{\max(G_{\rm var},0)}{s_E}
      +w_n\frac{e_1^2}{s_n^2},\qquad R=\frac1{1+L}.
\]

Default weights are one each; `energy_scale=sE=.01 Ha` and
`density_scale=sn=.01` are explicit workflow scales, not universal accuracy
thresholds. Callers can set different positive scales and nonnegative weights.
Negative numerical roundoff in `G_var` is suppressed only when assembling the
reward; its raw diagnostic is retained. The exact zero branch keeps the
physical second parameter derivative. An invalid candidate returns `value=0`,
`loss=NaN`, `valid=False`; raw energy/Fock data are preserved. Invalid concrete
references, including fractional occupations or excited Aufbau filling, are
rejected rather than converted to poor candidate scores.

The score is invariant to an additive energy constant and to an additive
constant potential `c*S` per fixed spin channel. Consequently it cannot assess
absolute functional energies: add the separate energy-label objective when
energy accuracy matters. A good score also does not establish candidate SCF
convergence, stability, or its final self-consistent density error. It measures
the fixed-reference/one-shot defect of the candidate only.

RKS, UKS and Hermitian complex GKS **operator assessment** use the shared
isolated-subspace eigensolver. Complex matrices are realified, doubling matrix
dimension and occupied rank; complete internal degenerate pairs are retained.
The external occupied/virtual gap must exceed `gap_tolerance` (default `1e-8 Ha`).
JIT and parameter gradients use subspace invariants rather than individual
eigenvectors. This does not add a spinor inverse solver or a missing custom
spinor energy adapter: direct GKS total-energy callbacks can be assessed, while
`energy_function` retains its documented adapter restrictions.

See `examples/iks/single_sample_reward.py` for a qualified cached CCSD/Wu-Yang
reference -> external analytic functional -> reward example. Input preparation,
JIT compilation, warm candidate evaluation and the AD/finite-difference check
are timed or reported separately; no new SCF or inversion is run. All candidate formulas, parameters, scales,
raw metrics and validity flags are saved to the example's JSON output.

The definition of the potential functional and its density-mismatch derivative
can also be inspected in the pinned [KSPies Wu-Yang implementation](https://github.com/ssnam92/KSPies/blob/cd99703ae588d3e081e97c8caec5c2aeff66d546/kspies/wy.py).

A failed offline inverse is not admitted as a reward reference. In small
Gaussian orbital spaces, a correlated density can require ensemble occupations;
a very flexible AO-product potential basis can drive the occupied/virtual gap
toward zero instead of producing a valid integer-occupation teacher. Gradient
convergence alone is insufficient. Retain the independent density and gap
checks, and use a qualified, balanced-potential-space reference for evaluation.
