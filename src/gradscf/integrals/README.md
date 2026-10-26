# Integral module ownership

`libcint` is the sole production Gaussian integral interface. The public entry
points are `prepare_basis`, `make_plan`, `make_auxiliary_plan` and
`backend_capabilities`. Ordinary SCF/DFT constructors and examples do not select
an integral engine. The former `cpu`, `native`, `jax` and `jax_reference` engine
names are rejected; CPU/GPU execution devices and JAX grid AO evaluation are
separate controls.

Static plans bind current JAX parameter arrays; they must not cache stale
exponents, coefficients or coordinates. The pure-JAX Gaussian recurrence
backend has been removed. JAX still owns normalization, AO/grid evaluation,
linear algebra, SCF response, and periodic FFT/GTH operations.

## Directory map

| Owner | Responsibilities |
|---|---|
| `basis/types.py` | Static shell topology, differentiable raw parameters and backend-independent shape validation |
| `basis/library.py`, `basis/data/` | Named/raw basis construction and complete bundled resources |
| `basis/cartesian.py` | Cartesian AO/shell views and padded grid payloads; no retired ERI quartet grouping |
| `basis/normalization.py`, `basis/contraction.py` | JAX normalization and primitive-to-contracted basis transforms |
| `basis/ecp.py`, `basis/auxiliary.py` | ECP data/adapter and auxiliary-basis specifications |
| `molecular/one_electron.py`, `molecular/eri.py` | Existing Cartesian operator entry points and packed AO-pair metadata |
| `molecular/jk.py` | JAX density contractions and libcint shell-direct products |
| `molecular/density_fitting.py` | Native auxiliary 2c/3c plans, fixed metric caches, whitening and projections |
| `molecular/factorization.py` | Spectral ERI factorization, distinct from auxiliary-basis DF |
| `grids/`, `periodic/` | Molecular grids/AO evaluation and periodic FFT/Coulomb/pseudopotential operations |
| `backends/native/packing.py` | Native table construction, normalization binding and storage checks |
| `backends/native/ffi.py` | Float64 CPU FFI value calls and shape/dtype validation |
| `backends/native/autodiff/evaluation.py` | Combined differentiable value dispatch and independent parameter storage |
| `backends/native/autodiff/` | Geometry, coefficient and exponent derivative products; shared batching |
| `_native/csrc/` | C++ shell drivers, integral products and immediate tangent/cotangent reductions |
| `_native/vendor/` | Pinned upstream sources, licenses and checksums; unchanged by this reorganization |

AO-to-MO transformations belong to the sibling `gradscf.ao2mo` package:
`blocks.py` handles selected dense, packed and DF blocks, `native.py` contracts
native shell slabs, `reference.py` builds complete small-system MO tensors,
and `response.py` supplies occupied/virtual response slices. Integral evaluation
and its derivative primitives remain in `gradscf.integrals`.

SCF configuration, XC/grid input assembly and RKS/UKS input payloads are owned
by `gradscf.scf.inputs`. Its grid cache lives with the grid implementation;
electron-count validation lives in `scf.core`. SCF implicit/explicit response
belongs to SCF/solver modules, not the integral backend. Existing public
integrals-level SCF-builder re-exports remain lazy and contain no implementation.

## Libcint execution and differentiation

`IntegralPlan` dispatches to native packing and mathematical derivative rules,
which call FFI and C++ kernels. Raw coefficients are normalized in JAX. C++
derivatives act on the normalized native coefficients and bare Gaussians; JAX
adds the normalization chain. Fixed auxiliary metric whitening is also JAX.

The C++ dispatcher delegates packed ERI, auxiliary DF, direct J/K and packed
J/K to separate translation units. Shared Coulomb shell traversal is in
`coulomb_shells.h`. Private paths such as `backends/native` identify implementation
ownership; they are not additional public engine choices.

Implemented product families have separate limits:

| Product family | Implemented AD boundary |
|---|---|
| Full Cartesian overlap, kinetic, nuclear attraction and ordinary Coulomb ERI | Coordinate products and adjoints through order four |
| Spherical overlap/kinetic/nuclear/ERI; dipole and grid-Coulomb `rinv` | First/second coordinate products; dipole/Rinv origin response is explicit |
| Packed s4/s8 ERIs and bounded slabs | First/second coordinate products without a full-ERI gather |
| Static ordinary Cartesian shell windows | Coordinate products through order four; spherical/range-separated higher products are separate unsupported cases |
| One-electron matrices and ordinary full-s1 ERIs | Raw coefficient first/second products and exponent first products, including the JAX normalization chain |
| Fixed-auxiliary RI | Orbital coefficient first/second and exponent first products with a precomputed, signature-checked metric |
| Shell-direct J/K | Density response and first/second coordinate response; coordinate AD requires zero screening cutoff |

Exponent Hessians, true higher coefficient products beyond the implemented
families, mixed coefficient/exponent/geometry derivatives, RI geometry/auxiliary
AD, and primitive basis-parameter AD of the direct J/K provider remain explicit
limitations. A projected contraction matrix keeps its own JAX dependence; it
does not create missing primitive-basis rules. Limits also depend on layout,
angular momentum and range parameter. No arbitrary-order claim follows from
the maximum order of one family.
See [_native/README.md](_native/README.md) for the exact numerical contract.

## Scalar ECP and storage boundaries

Gaussian/Coulomb operators use libcint. Scalar ECP values additionally require
the pinned PySCF ECP C subset in `_native/vendor/pyscf_ecp`; this distinct special
kernel is compiled into the private library, with its own license/provenance.
It is not a PySCF Python runtime or an alternative Gaussian engine. ECP metadata
and effective core charges must be preserved; unsupported ECP derivatives must
not be replaced by an all-electron Hamiltonian. Fixed-primitive ECP values plus
JAX contraction provide the existing coefficient path.

Full-s1 APIs explicitly materialize their requested four-index output. Packed,
shell-window, slab and direct paths retain their corresponding bounded storage;
their derivative limits must be checked independently. Fixed-primitive
four-index tensors are restricted to budgeted small-system validation, not a
general NNAO training cache. Working-array budgets do not constitute process-RSS
limits or account automatically for arbitrary external-model internals.

## Validation

Run from the repository root with CPU float64:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src python -m pytest -q \
  tests/integrals tests/test_integral_namespace.py tests/test_scf_inputs_api.py \
  tests/nnao/test_df_rhf_solver.py tests/nnao/test_native_exponent_scf.py \
  tests/nnao/test_joint_native_training.py
```

Build the private library with `python -m gradscf.integrals._native.build` before integral
tests. Basis resources and native sources must also be checked in a built
wheel. Remote experiments use their own frozen source/native snapshots; do not
overwrite those snapshots to apply a local module reorganization.
