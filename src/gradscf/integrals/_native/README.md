# Private native integral backend

This directory builds a private, serial CPU library with a JAX typed FFI entry
point. Runtime evaluation does not import or load PySCF. The source subset is
pinned to PySCF v2.13.0 and libcint v6.1.3; full commit IDs, original relative
paths, Git blob IDs and SHA256 checksums are in `vendor/manifest.json`.
Upstream Apache-2.0 licenses are retained at `vendor/pyscf/LICENSE` and
`vendor/libcint/LICENSE`. No vendor patches are currently needed.

## Directory layout

- `__init__.py`, `build.py`: private library loading and offline build CLI.
- `csrc/ffi.cc`: GradSCF C++ value FFI adapter.
- `csrc/geometry.cc`: streaming analytic coordinate JVP/VJP drivers.
- `csrc/tables.h`: shared table validation and owned libcint buffers.
- `vendor/`: pinned, unmodified upstream C sources, licenses and checksums.
- `include/`, `exports.*`, `CMakeLists.txt`: build configuration and ABI exports.
- `patches/`: documentation of upstream adaptations.
- `build/`, `lib/`: generated build products, excluded from Git and packaging.

Source distributions and wheels include the
native sources; compiled libraries are built locally for the installed JAX.

## Offline source build

From the repository root, with JAX, CMake >= 3.20 and a C99/C++17 compiler already
installed:

```sh
PYTHONPATH=src python -m gradscf.integrals._native.build
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q tests/integrals/test_native_values.py
```

The build verifies every vendored file before invoking CMake. It downloads
nothing and writes generated headers/object files under `src/gradscf/integrals/_native/build/`; the
private library goes into `src/gradscf/integrals/_native/lib/`. Both output directories
are ignored by Git. An installed editable package does not need `PYTHONPATH`.
Rebuild against the installed JAX headers after changing JAX versions.
Linux and macOS are supported build targets; no binary wheels are provided by
this initial source build. `--jobs` defaults to 2 for compilation only. Integral
evaluation is serial and does not modify process-wide threading settings.

All vendor symbols have hidden visibility and a linker export allowlist;
Only `GradSCFIntegrals`, `GradSCFGeometryJVP` and `GradSCFGeometryVJP` are exported. There is no dynamically
linked libcint, PySCF, BLAS, OpenMP, or Python library. Loading uses `RTLD_LOCAL`.

## Contract

```python
from gradscf.integrals.backends.native import evaluate
values = evaluate("overlap", atm, bas, env, nao, cart=True)
```

`atm` is a static NumPy int32 `(natm, 6)` array, and `bas` is a static NumPy
int32 `(nshell, 8)` array, using libcint's public table layout. The dynamic
`env` is a JAX float64 vector. The caller supplies normalized contraction
coefficients in libcint convention, coordinates in Bohr, and zero
`env[PTR_RANGE_OMEGA]`. Enable JAX x64 before constructing arrays. Tables may
be closed over by `jax.jit`; changing coordinates/exponents/coefficients in
`env` updates values without replacing the static topology.

| Operator | Shape | Meaning / units |
| --- | --- | --- |
| `overlap` | `(nao, nao)` | S, dimensionless for normalized AOs |
| `kinetic` | `(nao, nao)` | -1/2 Laplacian, Hartree |
| `nuclear` | `(nao, nao)` | Electron–point-nucleus attraction, Hartree |
| `dipole` | `(3, nao, nao)` | Position r relative to `env[1:4]`, Bohr; no electron charge factor |
| `eri` | `(nao, nao, nao, nao)` | Chemists' `(ij|kl)`, Hartree, full s1 tensor |

Both Cartesian and real spherical AOs use libcint/PySCF ordering and
normalization. This initial interface supports point nuclei, scalar Gaussian
orbitals with angular momentum 0..12, and ordinary Coulomb integrals. It
rejects nonfinite environments, nonpositive exponents, unsupported nuclear
models/range separation, invalid pointers and inconsistent AO counts.
ERI shell workspaces that exceed the upstream int32 indexing range are rejected
before allocation. The adapter owns the ERI workspace with checked C++ allocation
and invokes the unmodified PySCF `GTOnr2e_fill_s1` filling kernel.
This check includes a Cartesian contraction-cache lower bound even for spherical
output. Every native shell cache query must also return a strictly positive size;
libcint's zero-on-overflow sentinel is returned to JAX as an error before filling.

## Coordinate derivatives

`integrals.make_plan(..., backend="native").evaluate(...)` supports first-order
`jax.jvp`, `jax.vjp`, `jax.grad`, `jax.jacfwd` and `jax.jacrev` for nuclear
coordinates and basis centers, including JIT/batching. Dipole origins also
participate in AD; a default charge-center origin follows nuclear motion.
C++ evaluates analytic derivative shell blocks and immediately contracts them
with the tangent or cotangent. No full coordinate Jacobian or Python callback
is used. Nuclear-attraction operator motion and AO-center motion are separate.

The low-level raw-ENV `backends.native.evaluate` remains value-only: an arbitrary
ENV vector mixes geometry, exponents and coefficients. Use integral plans for
geometry AD. Native exponent/coefficient and higher derivatives remain
unsupported and fail explicitly. The JAX reference backend remains available
for those basis-parameter derivatives. GPU FFI, ECPs, periodic integrals,
spinors, range separation, packed ERIs and fused J/K are not exposed.
Full integral values still require O(nao^4) memory.

## Why these upstream files

PySCF's `pyscf/lib/gto/fill_int2e.c` provides the shell-to-full-tensor ERI
driver. Its only headers are libcint's `cint.h` and a local build `config.h`.
One-electron assembly lives in the GradSCF adapter and invokes libcint shell
kernels directly, avoiding PySCF's unrelated np_helper/BLAS dependencies.
Shared libcint optimizer objects also reference grid and two-/three-center
environment helpers; those small dependencies are included even though no
such operator is exposed. Large optional polynomial-fit tables, F12 code,
other generated operators, and upstream build/test machinery are excluded.
The pinned `autocode/grad1.c` and `autocode/grad2.c` provide the one-/two-electron
first-derivative kernels; their blob and SHA256 hashes are in the manifest.
All vendored files remain byte-identical to their pinned upstream blobs.

The numerical tests compare identical tables with PySCF at absolute tolerance
3e-11 and relative tolerance 3e-12, covering water/6-31g*, Cartesian and
spherical forms, eager/JIT values, and dynamic coordinate changes. A separate
process blocks all PySCF imports and checks a primitive overlap analytically.
Tests also verify unsupported AD and malformed-input rejection.
