# Compressed native integrals and auxiliary-basis RI

The molecular native backend can generate `s4`/`s8` ERIs directly, evaluate
shell-direct J/K, and generate two-/three-center Coulomb integrals. These
paths use the bundled libcint sources; they do not call PySCF at runtime.

## Exact integral paths

```python
from gradscf import integrals

top, parameters = integrals.prepare_basis(atom, "def2-tzvp", cart=False)
plan = integrals.make_plan(top)
eri = plan.evaluate("eri", parameters, aosym="s8")
j, k = integrals.build_jk_from_packed(eri, density)
# Or compute shell blocks and accumulate directly, without storing ERIs:
j, k = plan.get_jk(parameters, density, screening_threshold=0.0)
```

For real spatial AOs, `s4` stores a square lower-triangle AO-pair matrix and
`s8` stores its lower triangle. Both layouts use 64-bit pair addresses. The
native producer only allocates its packed result, libcint optimizer and shell
workspaces. The float64 CPU s8 consumer streams directly through the packed
buffer in C++; s4 and other devices use bounded JAX gathers. Both support
batched, nonsymmetric, complex densities, including generalized
spin-offdiagonal blocks, without restoring an N^4 array. The s8 operator's
bilinear JVP and density VJP reuse the native kernel; its ERI VJP uses bounded
JAX scatters. Mixed second derivatives are covered by regression tests.
Large s8 contractions use at most eight workers from the existing XLA thread
pool, with separate O(nao^2) accumulators and a fixed reduction order for a
fixed worker count; no additional threading runtime is required.

`get_jk` uses unique shell quartets and an optional density-independent
Schwarz threshold. Zero threshold adds no shell screening approximation.
Its native density operator has JVP/VJP and higher density derivatives.
Basis/exponent/geometry AD on this direct operator is explicitly rejected.

RHF/RKS, UHF/UKS, ROHF/ROKS and GHF/GKS integral solvers accept full, s4 and
s8 ERIs. Fixed-geometry native molecular assembly emits s4 directly; the
unrestricted payload's `eri` is therefore a packed pair matrix. Response
and neural energy consumers recognize this representation. Restricted
`mf.direct_scf()` uses the native direct provider. The UKS high-level direct
provider is not exposed yet; its exact packed route is supported.

Existing traced-coordinate assembly still uses the established dense native
geometry rule before packing the output. The new packed integral producer
and auxiliary integral producer are forward-only for integral parameters;
they do not silently discard coordinate or exponent derivatives.

## True RI/DF

```python
aux_top, aux_parameters = integrals.prepare_basis(
    atom, "def2-universal-jkfit", cart=False)
ri = integrals.make_auxiliary_plan(top, aux_top)
metric, three_center = ri.evaluate(parameters, aux_parameters)
packed_factors = ri.factors(parameters, aux_parameters)
```

The native calls compute `(P|Q)` and `(pq|P)` directly. The latter has shape
`(naux, nao*(nao+1)//2)`. Whitening uses the Coulomb metric's Cholesky factor;
if numerical linear dependence prevents Cholesky factorization, a spectral
projection drops eigenvalues below `lindep`. Factor construction is a host
setup/cache operation. `evaluate` itself is JIT-compatible.

`RKS.density_fit(auxbasis=...)` and `UKS.density_fit(auxbasis=...)` select this
path, defaulting to `def2-universal-jkfit`. The same auxiliary setting is
available in `RKSConfig`/`UKSConfig` for input assembly. Low-level
`jk_backend="df"` with no auxiliary basis retains the existing spectral
factorization of packed ERIs for compatibility; that is not auxiliary fitting.

## NNAO coefficient differentiation

With fixed primitive exponents and centers, the exact default is a projected
shell-direct operator. JAX forms `D_primitive = T @ D @ T.T`, the native
operator evaluates primitive J/K without storing ERIs, and JAX projects the
results with `T.T @ J_primitive @ T` and `T.T @ K_primitive @ T`. Native AD is
therefore needed only for the density map; coefficient and normalization
derivatives remain in JAX. The converged density is differentiated through the
implicit SCF fixed point.

The optional DF backend caches packed primitive factors and projects them
through the normalized contraction matrix:

```python
from gradscf.integrals.density_fitting import project_factors
contracted_factors = project_factors(primitive_factors, transform)
```

The projection evaluates `T.T @ B[Q] @ T` in auxiliary blocks; its JAX graph
includes coefficient normalization and supports zero initial coefficients.
Neither backend constructs a primitive or contracted four-index ERI.

Run the experiment with
`tools/optimize_methane_nnao.py --geometry FILE --jk-backend direct`, or select
`--jk-backend df --auxbasis def2-universal-jkfit`. DF factors remain in memory;
HDF5 out-of-core storage is not implemented by this interface yet.

RI is an approximation. Validate auxiliary-basis energy and coefficient-
gradient errors against exact contracted integrals. If centers/exponents
become training variables, derivatives of the native auxiliary integrals
must be added; reusing a fixed cache would not supply those derivatives.
