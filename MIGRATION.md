# Migrating from GradTDDFT to GradSCF

GradSCF is the current project and package name. This migration changes the
package identity and imports; it preserves the existing numerical methods,
function signatures, and scientific submodule names.

## DifferentiableSCF XC contracts

`DifferentiableSCF.run(molecule, functional, params)` retains the current Neural
XC/NeuralD/force-training interfaces. Historical SCF-only fallbacks were removed:

- Binding requires `bind_to_molecule_for_scf(params, molecule)`. Generic `bind`,
  `bind_to_molecule`, or an implicitly already-bound object is not substituted.
- Direct restricted potentials return seven entries: `(v_rho, v_grad, v_tau,
  v_lapl, kind, alpha, extra_fock)`. Old four/six-entry forms are rejected.
- Bound grid potentials use `grid_potential_components(molecule)`, returning
  the current three/four-entry form. Two-entry forms, `grid_potential`, and
  `local_potential` fallbacks are no longer accepted here. Grid gradients have
  shape `(ngrids, 3)`; transposed legacy arrays are not repaired automatically.
- Density-energy models expose `scf_xc_energy_and_alpha_for_density`, returning
  `(energy, alpha)` together. The old scalar-only callback and separate alpha
  fallback are not used by DifferentiableSCF.

The optimized direct-Fock path, SCF-specific bound path (including frozen
functional diagnostics), and eight-entry unrestricted potential path remain
active. Public response APIs on bound XC objects are unchanged. This cleanup
does not alter SCF iterations, density layouts, or implicit/unrolled semantics.

## SCF backward configuration

`gradscf.scf.SCFDifferentiationConfig` is shared by the functional/molecule SCF
adapter and the fixed-occupation orbital solvers. Use `mode="implicit"` or
`mode="unrolled"`; `impl` and `expl` remain accepted aliases. Pass this object
as `DifferentiableSCFConfig(differentiation=...)` or as the orbital solver's
`differentiation=...` argument. An explicit object takes precedence over the
legacy DFT backward fields. Existing DFT defaults are retained when it is absent.
Training workflows now honor their configured backward mode instead of forcing
implicit mode; the existing nonfinite-step recovery option may retry with implicit.

Orbital solvers now use JAX/Optax and return array-valued PyTrees, including
energy, convergence flags, and fixed-shape diagnostic histories. Convert arrays
to Python scalars only when reporting outside JIT/autodiff. The separate
`orbital-optimization` SciPy extra is no longer required.
Orbital updates use a Cayley retraction to preserve the metric while avoiding
large higher-order matrix-exponential derivative graphs; iteration trajectories
may differ from the former exponential parametrization.

The shared implicit policy requires a converged state by default and checks
the adjoint residual. Failed backward solves return nonfinite cotangents while
preserving forward diagnostics. Unrolled differentiates the finite iterations
and can be used before convergence. Fixed 0/1 occupations are static topology;
continuous integrals and overlap remain differentiable. Use
`orthonormalize_initial=True` to transport a fixed orbital seed as overlap changes.

The default implicit path now composes JVP/VJP rules through its root and linear
solves. `training.energy_and_forces`, `force_matching_loss`, and
`make_force_loss_and_grad` provide force supervision for differentiable energy
callbacks. Model/geometry mixed derivatives are supported with native integrals;
pure native coordinate Hessians and native basis derivatives remain unsupported.
Overlap orthogonalization now differentiates the inverse square root with a
Sylvester solve above the eigenvalue cutoff, avoiding spurious loss of response
at repeated overlap eigenvalues. Clipped overlaps retain regularized derivatives.

## SCF convergence, results and reference reuse

RKS now uses its public density tolerance and independent `conv_tol_grad=1e-7`,
matching the shared convergence policy. It no longer derives a gradient threshold
from `sqrt(conv_tol)` or relaxes the final level-shift check with an OR condition.
Runs may take additional iterations to satisfy the requested criteria.

`RKSResult` is the common internal PyTree; `TraceableRKSResult` remains an alias.
The eager entry retains Python scalar outputs. Cached RKS results/inputs are reused
for TD references; changing geometry or ground-state configuration requires
`kernel()` again. Preparing a reference preserves facade orbital layout/status.

Private test-reference helpers formerly imported from `gradscf.scf.features`
have moved to `tests/reference_scf_features.py`; no production compatibility
module is retained for those private helpers. HF wrappers and method-specific
RO/spinor logic remain separate.

## Name mapping

| Previous name | Current name |
| --- | --- |
| Distribution `td-graddft` | Distribution `gradscf` |
| `import td_graddft` | `import gradscf` |
| `from td_graddft import gto, scf, dft` | `from gradscf import gto, scf, dft` |
| `td_graddft_tools` | `gradscf_tools` |
| `src/td_graddft/` | `src/gradscf/` |
| `src/td_graddft_tools/` | `src/gradscf_tools/` |

Replace the package prefix in notebook imports, Python scripts, `python -m`
commands, dynamic import strings, and test monkeypatch targets. For example,
`td_graddft.scf.GHF` becomes `gradscf.scf.GHF`.

No `td_graddft` or `td_graddft_tools` compatibility namespace is shipped.
An old installation may still provide those names independently; importing it
does not select this checkout.

## Install or run the current checkout

With the required dependencies already available:

```sh
python -m pip install --no-deps -e .
python -c 'import gradscf; print(gradscf.__file__)'
```

For source-based execution without changing the environment:

```sh
PYTHONPATH=src python -c 'from gradscf import gto, scf, dft; print(scf.__file__)'
python -m pytest -q tests/test_gradscf_package.py
```

On c20, the working directory remains `/home/yjiao/GradSCF`, with the existing
`/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python` interpreter. The package
layout is now `src/gradscf`; the environment name does not need to change.

## Scientific names and existing data

Keep `tdscf`, `tddft`, TDA/TDDFT result names, and upstream GradDFT model or
adapter names: they identify methods or external projects. SCF class names
such as `UHF`, `ROHF`, `GHF`, `RKS`, `UKS`, `ROKS`, and `GKS` are unchanged.

The historical `reproducibility/v1.0.0/` files, their hashes, and third-party
basis data remain unchanged. NPZ target bundles retain the original embedded
metadata key so existing bundles remain readable. This does not provide
compatibility for arbitrary Python pickles containing old module paths.

The Git remote and repository URL still identify the existing GradTDDFT
repository; this API migration does not rename the GitHub repository.

## Consolidated integral modules

All integral implementations, basis resources and numerical grids now live
under `gradscf.integrals`. The following legacy files/exports are removed;
update imports rather than relying on compatibility aliases:

| Previous path | Canonical path |
| --- | --- |
| `gradscf.data.integrals` | `gradscf.integrals` |
| `gradscf.data.integrals.jax` | `gradscf.integrals.backends.jax_reference` |
| `gradscf.data.integrals.libcint.mol` | Removed; use native integral plans |
| `gradscf.data.integrals.libcint.autodiff` | Removed; native plans provide geometry AD |
| `gradscf.data.basis` | `gradscf.integrals.basis` |
| `gradscf.data.pyscf_basis_loader` | `gradscf.integrals.basis_data` |
| `gradscf.data.grid` / `data.grid_ao` | `gradscf.integrals.grids` / `integrals.grids.ao` |
| `gradscf.data.ris_auxbasis` | `gradscf.integrals.auxbasis` |
| `gradscf.scf.inputs` | `gradscf.integrals.assembly` |
| `gradscf._native` | `gradscf.integrals._native` |

Basis/grid functions previously exported from `gradscf.data` are exported from
`gradscf.integrals`. Public `gradscf.gto` facades remain available. All 321
original basis snapshot files and the supplementary JSON basis bundle are
retained; resource lookup now uses `gradscf.integrals.basis_data`. Angular
momenta, general contractions and kappa labels are preserved by the loader.
Loading data does not imply native ECP/spinor/operator support.

## External chemistry dependency removal

Production PySCF/gpu4pyscf bridges and object adapters are removed. Reference
converters now live in test-only `pyscf_data_reference`/`pyscf_adapters` helpers.
Comparative CLIs moved into `tests/comparisons/`. The `comparison-tests` extra
is the only dependency group that installs PySCF.

Use `init_guess="hcore"` or explicit density matrices. Legacy external guesses
are not silently approximated. Native full-ERI spectral factorization replaces
external auxiliary DF in production. Basis resource bytes are unchanged, but
Python-format resources now have the inert `.pydata` suffix.
