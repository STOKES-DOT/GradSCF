# Static molecular GW-BSE

`gradscf.bse` implements static GW-BSE for finite molecules, real closed-shell
orbitals and integer occupations. Both TDA and full BSE have matrix-free
Davidson paths and bounded dense reference paths. Singlet
and triplet energies, transition moments and length-gauge oscillator strengths
are supported. First-order response of isolated roots includes QP energies,
screening energies, factors, amplitudes and supplied dipoles. The numerical
eigenproblem and screening linear solves belong to `gradscf.solvers`.

This implements P0/P1 and real full-BSE forward/amplitude response from
[the development plan](../gw/BSE_PLAN.md). The dense reference is described in
[FULL_BSE_PLAN.md](FULL_BSE_PLAN.md); the matrix-free metric-response extension
is described in [MATRIX_FREE_BSE.md](MATRIX_FREE_BSE.md).
Open-shell/complex/periodic references, dynamic kernels and degenerate-cluster
properties are not exposed as implemented methods.

## Public workflow

```python
from gradscf import bse, dft, gto, gw

mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g")
mf = dft.RKS(mol, xc="hf", conv_tol=1e-12).run()
mygw = gw.GW(mf, nw=100).run()
response = bse.BSE(mygw, nroots=3, singlet=True).run()
print(response.e)                     # Hartree
print(response.oscillator_strength())  # dimensionless
```

`kernel()` returns `(energies, (X,Y))`; `run()` returns the object. Amplitudes
have shape `(nroots,nocc_window,nvir_window)`. TDA has `Y=0`; full BSE returns
both amplitudes. Request full BSE explicitly with
`bse.BSE(mygw, tda=False, solver="davidson", max_space=40).run()`.
Use `solver="dense", max_dense=256` for the independently validated reference.
The Davidson path does not materialize A or B and ignores the dense capacity
limit; it still obeys auxiliary/factor limits and its subspace capacity.
An explicit `BSEReference(qp_energy, screening_energy, mo_factors, nocc,
dipole_mo=...)` can replace the GW facade. These explicit inputs must describe
one common orthonormal orbital frame; they are not verified by running GW.

`occupied` and `virtual` select the optical window. `screening_occupied` and
`screening_virtual` independently select the screening window, defaulting to
all occupied/virtual orbitals. Omitting core transitions from the optical
window does not automatically remove their screening contributions.

GW-backed input uses G0W0's original MF spectrum for W0. No QP re-screening is
silently introduced. QP energies of every selected level must be covered by
both `qp_computed_mask` and `converged_mask`; the legacy GW result marks
unrequested MF-filled levels converged, so that flag alone is insufficient.
`GWResult.screening_energy` records the actual pole spectrum used in W by the
restricted/unrestricted CD drivers. A restricted result can be converted with
`BSEReference.from_gw_result(result, mo_factors=..., nocc=..., dipole_mo=...)`.
This preserves QP coverage/convergence masks and the recorded screening spectrum;
missing metadata is rejected. For converged evGW, that spectrum equals the final
QP spectrum, including any frozen unrequested MF levels. Factors and dipoles
must be in the returned orbital frame. This supports fixed-frame evGW output as
BSE input; it does not differentiate the evGW outer fixed point. qsGW/scGW
results without this provenance still require a separately justified explicit
reference and are not automatically adapted.

The eager interface checks source fingerprints. SCF/GW inputs, GW settings,
explicit arrays or BSE window/configuration changes invalidate cached results.
Failed reruns invalidate property access and reset convergence status. Missing
dipoles are completed through GradSCF integrals without repeating SCF; completing
an identical lazy dipole payload does not invalidate GW. Changing dipoles requires
a new BSE snapshot, while leaving the underlying QP calculation reusable.

## Kernel and normalization

For metric-whitened real factors, `(pq|rs)=sum_P L[P,p,q] L[P,r,s]`,

```text
Pi_PQ(0) = -4 sum_ia L_Pia L_Qia / (eps_scr[a]-eps_scr[i])
epsilon = I-Pi(0)
W[pq,rs](0) = L[:,p,q]^T solve(epsilon,L[:,r,s])
A[ia,jb] = (eps_qp[a]-eps_qp[i]) delta_ij delta_ab
           + kappa (ia|jb) - W[ij,ab](0)
kappa = 2 (singlet), 0 (triplet)
```

This uses **full W**, not the contracted W-v used by GW self-energy routines.
The exchange-like term uses bare Coulomb integrals; the attractive direct term
uses W. QP and screening spectra remain separate arguments. Zero-frequency
screening is the gapped imaginary-axis limit; no optical broadening parameter
enters its denominator. The spectral broadening helper in `tools.spectra` can
be used after the discrete excitations have been calculated.

Vectors obey `sum_ia (X[ia]**2-Y[ia]**2)=1`, reducing to unit X norm in TDA.
For singlets, `mu = sqrt(2) sum_ia (X[ia]+Y[ia]) dipole[ia]` and `f=(2/3)*Omega*sum_xyz mu**2`
in atomic units. Triplet electric-dipole transitions from a singlet ground state
have zero strength without SOC. This differs from the half-norm restricted
TD-SCF amplitude convention; existing TD-SCF property routines are not called
with incompatible BSE amplitudes.

The kernel convention follows the [2020 BSE perspective](https://arxiv.org/abs/2006.09440)
and the inspected [QuAcK GW-BSE driver](https://github.com/pfloos/QuAcK/blob/2236bfcda24ff0971358f5107636b506dd201bb1/src/GW/RGW_phBSE.f90).
See [REFERENCES.md](REFERENCES.md) for attribution and the distinction between
formula definitions and executed software comparisons.

## Functional differentiation

```python
space = bse.make_bse_space(nmo, nocc, occupied=occ_indices, virtual=vir_indices)
screening_space = bse.make_bse_space(nmo, nocc)
config = bse.BSEConfig(nroots=3, gradient_mode="implicit_eigenvector")

result = bse.run_bse(qp_energy, screening_energy, mo_factors, space,
                     screening_space=screening_space, config=config)
strengths = bse.oscillator_strengths(result, dipole_mo, space)
```

Construct static topology/configuration outside JIT. Explicit functional inputs
with omitted masks mean user-supplied energies; composition with a GW function
should pass its actual QP coverage/convergence masks. A fixed-rank factor input
can be differentiated without invoking the eager host spectral ERI factorization.

Default `gradient_mode="implicit_eigenvector"` supplies complete first-order
isolated-vector response. The optional energy-only mode keeps forward optical
values but invalidates their derivatives instead of returning only the energy
or dipole part. Screening parameter response is included. Neglecting the
functional derivative of W in deriving the standard static BSE kernel does not
mean stopping dW/dtheta when differentiating this specified model.

An extra root guards the selected spectrum. `response_valid` requires converged
guard/requested roots, positive selected frequencies and resolved internal and
boundary gaps. Exact/near degeneracies keep inspectable forward results but
invalidate individual-root derivatives. TDA cluster projector observables are
a later interface; there is no implicit root-count expansion or gap broadening.
In TDA, `stable` denotes positive selected frequencies, not an independent
global stability certificate for the starting mean field. Negative TDA roots
are retained as diagnostics and rejected for optical property evaluation.

Full BSE checks `A-B` and `A+B`, with absolute stability tolerance 1e-10
Hartree. In dense mode, `stability_margins` reports their minimum eigenvalues
and a passing result has `stability_certified=True`. In Davidson mode these
are **lowest Ritz estimates**, with `stability_residual_norms`; positive,
converged estimates set `stable=True` but `stability_certified=False`. They
are numerical screening, not a rigorous global proof that no lower unstable
mode was missed. Inspect both fields when certification matters. TDA reports
no global certificate and `stability_margins=None`.

Dense full-BSE vectors use Cholesky-Hermitian reduction; Davidson uses a
structure-preserving projected Hamiltonian/metric pencil and a metric-constrained
implicit amplitude solve. Nonpositive/unresolved stability checks give NaN
physical outputs and false convergence/response status. No imaginary mode is
converted into a real excitation. `response_valid` certifies the primal root
and gap prerequisites; a subsequent failed adjoint solve still returns NaN.
A full-BSE request containing any unresolved internal/boundary degeneracy
invalidates derivatives for the **whole requested root set**, consistently for
JVP and VJP. A smaller isolated prefix can still be differentiated. Forward
values for stable degenerate roots remain available.

The verified contract is first-order fixed-MO response and a G0W0+BSE composition
on a valid QP branch. It is not a full nuclear/basis derivative, a full evGW/qsGW
fixed-point derivative, a Matsubara-to-real-axis continuation, or a root-tracking
rule across GW residue/pole crossings.

## Storage and solver boundaries

The default Davidson path does not form the transition-space A matrix or a
four-index W tensor. Screened virtual-pair factors are prepared in slabs and
the direct-kernel action is accumulated in auxiliary blocks. Static blocks are
used so the action supports the linear transpose needed by the common adjoint.
There is still a dense auxiliary dielectric and stored full MO/screened factors.

Defaults: `max_aux=1024`, `max_factor_elements=20_000_000`, `max_space=40`,
`block_size=16`. The first two are checked before the GW facade transforms AO
factors to MO factors. They are capacity controls, not measured peak-memory
bounds; screening, factors, workspaces and AD intermediates coexist. The legacy
GW calculation before this adapter has its own storage requirements.

`solver="dense"` is a bounded oracle with `max_dense=256`, checking capacity
before materializing A/B. Its stable RPA reference uses n-by-n matrices rather
than a doubled physical matrix. Full-BSE Davidson instead allocates paired
bases proportional to `ntrans*max_space` and projected matrices bounded by
`2*max_space`. Its metric adjoint uses checked diagonally preconditioned GMRES.
The capacity must fit the requested roots, an available guard root and two
correction slots (unless it already spans the full transition dimension).
Increase `max_space` or `max_cycle` if a requested or guard root fails; there
is no hidden dense fallback or automatic relaxation of tolerances. Davidson requires
space for the requested roots plus its guard root and uses deterministic
full-support guesses (seed 0 by default). Shared direct linear solves now accept
multiple RHS columns, allowing one factorization per screening slab with checked
implicit and transposed solves. Large-auxiliary matrix-free screening, true RI
versus spectral-ERI-factor benchmarks, GPU scaling and out-of-core operation
remain future work.

Run [water_tda.py](../../../examples/bse/water_tda.py) or
[water_full.py](../../../examples/bse/water_full.py) for native examples.
Executed checks and limitations are in [VALIDATION.md](VALIDATION.md).
