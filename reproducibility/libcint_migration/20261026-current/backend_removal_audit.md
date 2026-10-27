# Production integral removal audit

This audit compares the source snapshot `7d2e859` with this worktree. The
original checkout is not a migration target and remains unchanged.

## Removed production implementation

The six tracked files in `src/gradscf/integrals/backends/jax_reference/` are
deleted: `__init__.py`, `_common.py`, `one_electron.py`, `two_electron.py`,
`direct_jk.py`, and `screening.py`. Their Gaussian recurrence code is retained
only in Git history, not in a legacy/experimental package.

`IntegralPlan`, molecular matrix/element wrappers, SCF input assembly and
direct-J/K consumers now use the same libcint engine. The old JAX dispatch,
SCF Cartesian-provider fallback, retired precompile exports, and quartet
group/scatter metadata used only by that engine are removed. Dedicated tests
of the removed implementation are retired; physical normalization, finite
differences, parameter/geometry response, and independent PySCF oracles remain.

`make_plan` and `backend_capabilities` reject `cpu`, `native`, `jax`,
`jax_reference`, and `pyscf`; those names are not aliases. Ordinary molecular
HF/DFT constructors no longer accept an integral-engine selector. Execution
device, J/K representation, and JAX AO-grid settings remain separate controls.
Neural-XC workflow settings and existing example/test consumers are migrated.
The public basis spectral-factorization helper also no longer forwards its
obsolete `engine` argument.

Native library absence/staleness raises with the build command. There is no
automatic Gaussian fallback, zero derivative policy, or Python `mol.intor`
bridge in the production calculation chain.

## Deliberately retained

* JAX normalization, projection/contraction, grid/AO evaluation, XC, array
  algebra, response solvers, and periodic FFT/GTH code are not duplicate
  Gaussian integral engines.
* Full, packed, shell-window/slab, direct J/K and true auxiliary 2c/3c RI
  are products of the same libcint engine, not competing backends. Spectral
  compression of an existing 4c array remains separately named and is not RI.
* `_native` and `backends/native` are private implementation directories.
  Existing provider/class/checkpoint names are not renamed unnecessarily.
* Pinned PySCF filling-driver C and scalar-ECP C are necessary, licensed
  native components. ECP is not claimed to be a libcint Gaussian kernel.
  Effective charges/core counts and qvszps family data remain intact.
* Bundled PySCF basis resources and licenses remain. Their aliases and literal
  element blocks are parsed as data with AST; upstream Python imports in
  `.pydata` resources are not executed.
* Independent PySCF/MOLGW comparisons and the optional
  `gradscf_tools.cis_cluster_reference` oracle remain external references.
  They do not provide inputs to the GradSCF production path. PySCF is only an
  optional comparison dependency.

## Build and runtime evidence

`upstream_verification.json` records the direct upstream archive URL, SHA256,
commit `c72d30662ba08048940c9000cc352a220c867712` (libcint 6.1.3), and bytewise
equality of all 45 selected libcint files. Vendor files were not edited.

`evidence/production-wheel.log`, `production-install.log`,
`build-installed-production.log` and `no-pyscf-smoke.log` record a fresh wheel,
installation into `env-production`, and a separate extension build there.
The smoke asserts `find_spec('pyscf') is None`, and that both the module and
actual loaded library reside under that environment. RHF/PBE/MP2 and NNAO
coefficient/exponent forward, pullback and parameter update pass.

The production and comparison environments share immutable dependency files
through explicit symlinks; no original editable pointer or environment is
modified. `.pth`, editable links, credentials, caches and training data are
not copied. The comparison environment alone exposes PySCF.

## Explicit limits

No arbitrary-order AD claim is made. Primitive ECP derivatives, general
auxiliary/RI geometry derivatives, exponent Hessians, and true mixed basis /
geometry products remain unsupported where they were unsupported before.
The contracted variable-exponent RI projection now explicitly rejects ECP
instead of silently omitting it; fixed-primitive ECP contraction remains.
Existing GPU model/array paths use CPU libcint products with explicit transfers,
not a GPU-native Gaussian integral engine.

## Final acceptance

`evidence/s9/installed-final-s9-smoke.log` and
`installed-source-proof.json` verify the final no-PySCF prefix and actual loaded
library, with486 code/resource hashes matching the current worktree. The
separate installed first-call group passes12 bounded derivative-only requests
without any earlier value call or diagnostic preloading. Its handler-init
regression is fixed in the production geometry-family builder.

The retired `jax_reference` directory is absent in the working filesystem as
well as Git-tracked source. The obsolete ordered-quartet helper is removed,
leaving one maintained shared metadata scheduler for the new derivative paths.
Historical immutable comparison snapshots remain outside the production tree.
No source engine was renamed into a legacy or backup package.
