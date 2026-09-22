# GW+BSE development proposal

Date: 2026-09-22. Repository inspected at `a924509` on `release/v1.0.0`.
Status: P0/P1 real closed-shell static TDA implementation completed and tested
in `feat/molecular-bse`. See the [implemented API](../bse/README.md) and
[executed validation](../bse/VALIDATION.md). The next increment adds bounded stable full BSE and fixed-frame GW-result
screening provenance; see [its scope](../bse/FULL_BSE_PLAN.md). Scalable full
BSE and remaining later-stage items below remain proposals.
Scope decision: the user selected finite molecules and closed-shell references
as the first priority on 2026-09-22.

The confirmed first target is finite molecules with real closed-shell
orbitals, integer occupations and a static particle-hole GW-BSE kernel.
Start with singlet/triplet TDA, then full BSE. Unrestricted, periodic, dynamical
and finite-temperature scGW-based response are explicit subsequent stages.

## Reference strategy

| Reference | Role in this project |
| --- | --- |
| Blase, Duchemin, Jacquemin and Loos, 2020 perspective | Static GW-BSE kernel, spin factors and approximation boundaries; equations 17–23 |
| QuAcK | Inspectable molecular A/B kernel and separate screening/QP spectra; small-system formula oracle |
| MOLGW | Primary external molecular GW+BSE energy/oscillator-strength comparison using Gaussian orbitals |
| VOTCA-XTP | Independent molecular comparison and matrix-free/RI implementation design |
| BerkeleyGW | Later periodic, momentum-transfer, Coulomb-head and optical-limit reference |
| Recent online PySCF BSE | Additional candidate oracle after pinning its exact revision and conventions |

Sources consulted:

- [Blase et al., arXiv:2006.09440v2](https://arxiv.org/abs/2006.09440), published as
  [J. Phys. Chem. Lett. 11, 7371–7382 (2020)](https://doi.org/10.1021/acs.jpclett.0c01875).
- QuAcK [GW-BSE driver](https://github.com/pfloos/QuAcK/blob/master/src/GW/RGW_phBSE.f90),
  [resonant kernel](https://github.com/pfloos/QuAcK/blob/master/src/GW/RGW_phBSE_static_kernel_A.f90),
  [coupling kernel](https://github.com/pfloos/QuAcK/blob/master/src/GW/RGW_phBSE_static_kernel_B.f90).
- [MOLGW BSE tutorial](https://www.molgw.org/tuto_bse/),
  [MOLGW capabilities](https://www.molgw.org/).
- [VOTCA-XTP architecture](https://www.votca.org/xtp/Architecture.html).
- [BerkeleyGW BSE tutorial](https://berkeleygw.org/documentation/tutorial/silicon-bethe-salpeter-equation/).
- [Current online PySCF BSE source](https://pyscf.org/_modules/pyscf/gw/bse.html).

Local PySCF **2.9.0** has no `pyscf.gw.bse` module (checked with importlib).
The online documentation now exposes a newer BSE implementation; this does not
establish its availability in the installed version or numerical agreement.
It is useful as another reference, not the sole acceptance oracle. No external
BSE executable was installed or run in the original planning stage. The P0/P1
implementation subsequently compiled and ran two pinned QuAcK Fortran kernels;
this is a kernel oracle, not a complete external molecular GW+BSE calculation.
Before generating further
reference artifacts, pin software revisions, basis assets and input conventions.
Keep theoretical citations, adapted-source attribution and test-oracle provenance
separate, following the existing CI/CC documentation convention.

## Repository audit: reusable pieces and gaps

| Existing component | Reuse / required change |
| --- | --- |
| `gw/polarizability.py::rho_response_iw` | Reuse the metric-whitened auxiliary polarizability, evaluated at zero frequency |
| `gw/screened.py` | Existing self-energy routines return contracted **W-v**; add a reusable full static-screening action without changing their semantics |
| `gw/g0w0.py`, `gw/qp.py` | G0W0 QP energies and existing conditional first-order QP response |
| `gw/types.py::GWResult` | Add actual QP-computation coverage/provenance; current unrequested orbitals contain MF energies and convergence flags True |
| `gw/rgw.py::GW` | Add checked source snapshots and a stable BSE-reference adapter; avoid coupling BSE to a private `_df_factors()` method |
| `df/jk.py` | Reuse factors/contractions; distinguish true auxiliary-basis RI from spectral decomposition of packed ERIs |
| `solvers.solve_hermitian` | TDA dense oracle and matrix-free Davidson; isolated-root energy/vector response |
| `solvers.solve_spectral_projector` | TDA complete-cluster observables at internal degeneracy; no generic full-BSE metric response yet |
| `solvers/eigen/rpa.py` | Reuse structured full-BSE forward/eigenvalue kernels; current X/Y are stopped, so property AD needs additional work |
| `tools/spectra.py` | Reuse spectral broadening; extract or adapt dipole contractions with an explicit normalization conversion |

Current evGW and qsGW outer fixed points are eager-only: their complete
self-consistency AD is not available. Their validated outputs can eventually
be BSE inputs, but this must not be advertised as end-to-end differentiable
evGW/qsGW+BSE. Current matrix Matsubara scGW returns `mo_energy=None`; its
static Fock spectrum is not a set of interacting QP poles. A scGW-to-BSE path
requires a separately defined spectral reduction/continuation or a genuine
finite-temperature two-particle formulation.

## Mathematical contract

Use an orthonormal real spatial MO basis, occupied indices i,j and virtual a,b.
The internal transition order is explicitly `(i,a)`, not inferred from array
shape. Energies are Hartree; position integrals are Bohr.

Define metric-whitened factors by `(pq|rs) = sum_P L[P,p,q] L[P,r,s]`.
For a closed-shell independent-particle screening reference,

```text
Pi[P,Q](0) = -4 sum_(i,a in screening space)
                  L[P,i,a] L[Q,i,a] / (eps_scr[a] - eps_scr[i])
epsilon_aux = I - Pi(0)
W[pq,rs](0) = L[:,p,q]^T solve(epsilon_aux, L[:,r,s])
```

This is **full W**, including the bare interaction. The existing GW quantity
`solve(I-Pi,Pi)` is only its correlation part in the auxiliary representation.
Static screening uses the imaginary-axis zero-frequency limit, with positive
screening gaps checked explicitly; no artificial broadening is needed for this
gapped molecular limit. GW self-energy broadening and optical plot broadening
are separate controls.

The static spin-adapted matrices are

\[
A_{ia,jb}=(\epsilon_a^{QP}-\epsilon_i^{QP})\delta_{ij}\delta_{ab}
           +\kappa(ia|jb)-W_{ij,ab}(0),
\]
\[
B_{ia,jb}=\kappa(ia|bj)-W_{ib,aj}(0),\qquad
\kappa=2\;(S=0),\;0\;(S=1).
\]

These conventions follow the cited perspective and are cross-checkable against
the QuAcK kernels. The exchange term uses the **bare** Coulomb interaction;
the attractive direct term uses screened W. Screened integrals must not be
replaced by the self-energy's already-contracted W_mn arrays.

TDA solves `A X = Omega X`. Full BSE solves

\[
\begin{pmatrix}A&B\\-B&-A\end{pmatrix}
\binom{X}{Y}=\Omega\binom{X}{Y}.
\]

Use `X.T X=1` in TDA and `X.T X-Y.T Y=1` in full BSE. For a closed-shell
singlet and one-electron dipole matrix d, `mu = sqrt(2) sum_ia (X+Y)[ia] d[ia]`
and `f = (2/3) Omega sum_xyz |mu|^2` in atomic units. Triplet electric-dipole
strengths from a singlet ground state vanish without spin-orbit coupling.
The existing restricted TD-SCF spectral helper uses a factor 2 with its own
amplitude normalization; a BSE result cannot be passed to it unmodified.

The first model uses static W and the usual GW-BSE kernel approximation that
omits the functional derivative of W in deriving the kernel. **That does not
mean stopping dW/dtheta in AD of the implemented model.** Those are different
derivatives and different approximations. No additional QP Z weights are inserted
into the first model's BSE kernel or transition normalization without a separately
specified and validated variant.

## Energy, orbital and window provenance

Keep three concepts distinct:

1. `qp_energy`: sets electron-hole gaps.
2. `screening_energy` and screening orbitals: define Pi and W.
3. `excitation_space`: defines the transitions retained in A/B.

For G0W0+BSE, default screening is W0 from the same MF spectrum/orbitals as GW;
inserting QP energies into screening is a distinct prescription, not a silent
default. For evGW, screening must follow the actual updated/frozen-W policy.
For qsGW, updated orbitals require consistently transformed factors and dipoles.
QuAcK's separate `eW` and `eGW` arguments are a useful reference for this boundary.

The screening window and optical excitation window must be independent. A small
optical window does not imply truncating high virtual states out of Pi. Frozen
core choices must state which of these calculations they affect.

Add an explicit `qp_computed_mask` (or equivalent retained index list) to the
GW return/adapter contract. BSE must verify QP coverage and convergence for every
selected occupied/virtual level. Current `converged_mask` and zero residuals
cannot identify uncomputed orbitals. Never silently treat an MF-filled level
as a completed GW correction. Explicit scissor or supplied-QP models can be
supported as separately labeled inputs.

Bind QP spectrum, occupations, orbital frame, integral-factor convention,
screening prescription/window and source signatures in a validated reference
snapshot. Reject stale or incompatible snapshots. Restricted molecular inputs
must not accidentally accept UHF, fractional occupations, complex orbitals or
k-point arrays through shape coercion.

## Proposed module ownership and API

```text
src/gradscf/bse/
  __init__.py       public exports
  types.py          BSEConfig, BSEReference, BSESpace, BSEResult
  reference.py      checked GW/explicit-input adapters and provenance
  space.py          transition and screening index spaces
  kernel.py         factorized A/B actions and bounded dense assembly
  response.py       physical problem orchestration through shared solvers
  properties.py     dipoles, oscillator strengths; later NTO/exciton analysis
  api.py            BSE facade, kernel/run and property methods

src/gradscf/gw/screened.py
  reusable static auxiliary screening state/action

src/gradscf/solvers/
  shared screening linear solves, Hermitian/RPA forward and backward rules
```

No eigensolver, Davidson, GMRES or numerical AD-rule copy belongs in `bse`.
`response.py` owns assembly and interpretation, not numerical iteration.
The module is separate from `gw` because it represents a two-particle response;
it is also separate from `tdscf`'s density-functional kernel implementation.

Implemented TDA facade:

```python
mf = dft.RKS(mol, xc="hf").run()
mygw = gw.GW(mf).run()
mybse = bse.BSE(mygw, tda=True, singlet=True, nroots=5).run()
print(mybse.e)
print(mybse.oscillator_strength())
```

The functional JAX path accepts explicit QP energies, MO factors, static
screening state and static transition topology. It must not invoke an eager SCF
facade from inside JIT. `BSEResult` should include energies, X/Y, residuals,
per-root convergence, stability status, response validity, normalization and
window metadata. Inability to certify global stability is distinct from a
certified instability. Properties must check whether amplitude response exists.

## Differentiation contract

| Quantity | First supported contract |
| --- | --- |
| TDA isolated excitation energy | First-order JVP/VJP via shared Hermitian solver |
| TDA isolated-state oscillator strength | Include eigenvector, dipole, QP and screening response |
| Complete isolated TDA root cluster | Projector-based observables and eigenvalue sums, including internal degeneracy |
| Full-BSE excitation energy | Existing metric-normalized response, after stability/normalization validation |
| Full-BSE oscillator-strength gradient | Requires new shared X/Y response; forward values alone do not establish this |
| G0W0+BSE composition | Explicit functional composition on a valid QP branch with differentiable inputs |
| evGW/qsGW+BSE outer response | Deferred until their full fixed-point derivatives exist |
| scGW+BSE and nuclear/basis gradients | Separate upstream spectral/kernel/backend work |

For TDA, a complete-cluster transition-strength sum is expressed through P.
With spin-normalized dipole probe d, the energy-weighted oscillator-strength
sum is proportional to `d.T @ P @ A @ P @ d`, not merely `d.T @ P @ d` for a
cluster containing different energies. Reuse `solve_spectral_projector` actions
and require a gap to the excluded spectrum. Do not promise derivatives of an
arbitrary labeled vector inside a degenerate cluster.

Full BSE has an indefinite metric and is outside the current Euclidean projector
contract. A stable Hermitian reduction through A-B can be investigated, but its
metric/matrix-function response must be included. Imaginary modes, negative
metric roots and instabilities must not be repaired by clipping, absolute-value
square roots or silently discarding problematic roots.

## Memory and computation

Use factorized kernel actions from the start, with dense A/B restricted to a
small explicit oracle limit. Block the screened direct contraction as well as
the MO-factor transforms. Budget the auxiliary dielectric matrix, MO factors,
Davidson vectors, work arrays and AD intermediates separately. Matrix-free BSE
alone does not make an naux-by-naux screening inverse inexpensive.

For the existing Cartesian cc-pVDZ BODIPY setup (245 orbitals, 49 occupied),
the full optical space has `49*196=9604` transitions. At float64, one dense A
contains 0.738 decimal GB and the explicit doubled full-BSE matrix 2.952 GB,
before work arrays and gradients. These are payload calculations, not measured
peak memory. This motivates bounded dense tests and a matrix-free production API.

The existing packed-ERI spectral factorization is not automatically a small
auxiliary RI basis. Its host NumPy path is also not a differentiable integral
factorization. The first AD contract can take fixed-rank factors as inputs;
end-to-end factor/basis response and rank-threshold changes need separate tests.
If large auxiliary spaces require CG or other numerical solvers, implement them
in `gradscf.solvers` only, with their own forward/backward validation.

## Validation gates and delivery order

| Stage | Deliverable | Acceptance gate |
| --- | --- | --- |
| P0 | Conventions, reference snapshots, static W action and QP-coverage metadata | Independent static dielectric/pole-expansion agreement; no W/W-v or frame ambiguity |
| P1 | Real molecular singlet/triplet TDA; dense oracle plus factorized Davidson; optical properties | Same-input A/actions/roots/dipoles against independent NumPy and a pinned external kernel |
| P2 | Full BSE forward plus excitation-energy response | Dense versus shared structured solver, metric normalization, B=0 limit and instability diagnostics |
| P3 | Property AD and degenerate TDA cluster observables; full-BSE amplitude-response extension | Reconverged finite differences and gauge-invariant cluster tests; explicit unsupported boundaries |
| P4 | Broader GW adapters, blocking and practical memory controls | Screening-policy/frame tests and reproducible basis/window convergence sweeps |
| P5 | UHF/ROHF, periodic and dynamic/scGW extensions as separate increments | Spin, momentum, Coulomb-limit and model-specific external oracles |

The recommended first implementation increment is **P0+P1**. P1 includes
isolated-root TDA energy/vector response; P3 completes broader property/cluster
contracts rather than excusing stopped-vector property gradients in P1.

Validation should proceed at three distinct levels:

1. **Algebraic:** explicit loop-built kernels versus factorized actions; symmetry,
   one-transition analytic solution, independent gaps when the kernel is off,
   HF energies plus W=v recovering CIS/TDHF, B=0 recovering TDA, triplet dipole
   selection rule, and normalization/contraction checks.
2. **Same-input external oracle:** identical factors or four-index integrals,
   QP energies, screening energies, occupations and dipoles. Compare A/B before
   eigenvalues and invariant strengths. This isolates BSE errors from GW/SCF
   differences. Use QuAcK and/or MOLGW with pinned revisions; newer PySCF is an
   additional candidate. FCI/CC excitation energies are physical comparisons,
   not exact BSE numerical oracles.
3. **End-to-end molecules:** H2, LiH, H2O and ethylene first; larger chromophores
   including BODIPY only after the memory/window tests. Match geometry, AO/aux
   bases, spherical/cartesian conventions, starting functional, QP solver,
   frozen spaces, W prescription, frequency grids and broadening definitions.

Initial target tolerances (to be confirmed by convergence studies, not claimed
results): about 1e-10 Hartree for small fixed-input kernel/action comparisons,
1e-8 Hartree for same-input roots, 1e-7 absolute strength differences, and
1e-6 or tighter directional-gradient agreement using several FD step sizes.
End-to-end external spectra get a separate tolerance budget after QP/grid/basis
errors are resolved; do not require accidental bitwise agreement between codes.

Record backend, dtype, seed, dimensions, memory controls, exact command, elapsed
time, peak-memory measurement method and all numerical errors with artifacts.
The original plan did not itself establish numerical results. Executed checks
are recorded separately in [VALIDATION.md](../bse/VALIDATION.md).
