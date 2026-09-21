# GradSCF

**Differentiable electronic structure in JAX.**

GradSCF is a Python framework for molecular and periodic electronic-structure
calculations, differentiable numerical methods, and machine-learned models.
It brings together Hartree–Fock and density-functional theory, excited-state
response, configuration interaction, coupled cluster, and GW, with tools for
learning exchange–correlation functionals and atomic-orbital basis sets.

[Installation](#installation) · [Quick start](#quick-start) ·
[Methods](#supported-methods) · [Differentiation](#differentiation-and-numerical-solvers) ·
[Examples](#documentation-and-examples) · [Citation](#citation)

## Highlights

- **Molecular and periodic calculations.** Mean-field, response, and correlated
  methods with explicit reference-state and backend conventions.
- **Differentiable numerical solutions.** Shared eigensolvers, linear solvers,
  and fixed-point response rules connect physical equations to JAX autodiff.
- **Learnable electronic-structure models.** Neural XC functionals and
  MACE-conditioned Gaussian basis contractions can enter self-consistent workflows.
- **Python interfaces at two levels.** PySCF-style calculation objects for
  interactive use, and functional kernels for JIT compilation and differentiation.

## Installation

Install from source on Linux or macOS. Use Python 3.11 or newer for repository
tests; the package declares Python 3.10+ support. The native CPU integral backend
requires CMake 3.20+, a C99/C++17 compiler, and a BLAS library (Accelerate on macOS).

```bash
git clone --branch release/v1.0.0 https://github.com/STOKES-DOT/GradSCF.git
cd GradSCF
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m gradscf.integrals._native.build
python -c "from gradscf import gto, dft, ci, cc; print('GradSCF imported')"
```

Native sources are included in the repository; the build does not download
upstream code. Rebuild the extension after changing JAX versions. See the
[native build instructions](src/gradscf/integrals/_native/README.md) for details.

Optional dependencies can be added to the same environment:

- `python -m pip install -e ".[upstreams]"` — the `jax-xc` backend for XC functionals.
- `python -m pip install -e ".[dev,comparison-tests]"` — pytest and PySCF comparisons.
- `python -m pip install -e ".[nnao]"` — the pinned upstream MACE-JAX dependency.
- `python -m pip install -e ".[reproducibility]"` — plotting, HDF5, and evaluation dependencies.

The native integral runtime and CI/CC kernels do not require PySCF. Comparison
tests and optional PySCF bridges have separate requirements. JAX device support
does not imply that every GradSCF backend runs on that device: the native
integral extension is a CPU implementation.

## Quick start

The following examples run independently after building the native backend.
Set `JAX_PLATFORMS=cpu` before starting Python; each example enables float64
before constructing arrays. Molecular coordinates below are in Angstrom and
energies are in Hartree.

### Run a ground-state calculation

Use `RKS(xc="hf")` for the closed-shell HF facade, then correlate its reference
with CCSD. The same mean-field object can also supply CI and response calculations.

```python
import jax
jax.config.update("jax_enable_x64", True)
from gradscf import gto, dft, cc

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom")
mf = dft.RKS(mol, xc="hf", conv_tol=1e-12).run()
assert mf.converged
mycc = cc.CCSD(mf).run()
assert mycc.converged
print("HF:", float(mf.e_tot), "CCSD:", float(mycc.e_tot))
```

For open-shell references, use `scf.UHF` or `scf.ROHF`.
`cc.CCSD(mf)` and `ci.CISD(mf)` dispatch according to the reference; explicit
`UCCSD` and `UCISD` interfaces are also available. See the
[open-shell example](examples/cc/open_shell_ground.py).

### Compute excitation energies

This HF example evaluates singlet TDA excitation energies. The `tdscf` facade
also provides full TDHF/TDDFT for supported references and XC kernels.

```python
import jax
jax.config.update("jax_enable_x64", True)
from gradscf import gto, dft, tdscf

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom")
mf = dft.RKS(mol, xc="hf", conv_tol=1e-12).run()
assert mf.converged
td = tdscf.TDA(mf, nstates=1)
td.kernel()
assert td.converged
print("Excitation energies / Ha:", td.e)
```

### Differentiate a calculation

Here a scalar scales the two-electron MO integrals while the orbital basis
remains fixed. The derivative includes the response of the converged CC
amplitudes. It is an integral-parameter derivative, not a nuclear force.

```python
import jax
jax.config.update("jax_enable_x64", True)
from gradscf import gto, dft, cc
from gradscf.scf.reference import reference_from_source

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom")
mf = dft.RKS(mol, xc="hf", conv_tol=1e-12).run()
ref = reference_from_source(mf)
config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)

def energy(coupling):
    result = cc.run_cc(ref.h1, coupling * ref.eri, nocc=ref.nocc,
                       nuclear_repulsion=ref.nuclear_repulsion, config=config)
    return result.total_energy

value, derivative = jax.jit(jax.value_and_grad(energy))(1.0)
print("Energy / Ha:", float(value), "dE/dcoupling:", float(derivative))
```

## Supported methods

Forward availability and derivative coverage are separate. Follow the linked
module documentation for reference restrictions, inputs, and validated paths.

| Family | Available methods | Scope and documentation |
| --- | --- | --- |
| Molecular mean field | RHF, UHF, ROHF, GHF; RKS, UKS, ROKS, GKS; UHF/UKS stability analysis | [SCF](src/gradscf/scf) and [DFT](src/gradscf/dft) interfaces; closed-shell HF facade uses `RKS(xc="hf")` |
| Periodic mean field | Gamma/k-point HF and DFT, GTH pseudopotentials, FFT density fitting, bands | [Periodic modules](src/gradscf/pbc); [example](examples/periodic_h2.py) |
| Excited-state response | TDA and full TDHF/TDDFT, transition properties and spectra; periodic q=0 response | [Molecular facade](src/gradscf/tdscf), [periodic response](src/gradscf/pbc/tdscf.py); reference/kernel restrictions apply |
| Configuration interaction | Restricted singlet/triplet CIS, singlet CIS(D), spin-conserving UCIS; CISD/CISDT/CISDTQ and general rank truncation | [CI guide](src/gradscf/ci/README.md); real molecular determinant spaces from RHF/UHF/ROHF |
| Coupled cluster | Restricted CCS/CCD/CCSD/CC2/LCCD/LCCSD, CCSD(T), CCSD+T(CCSD), Lambda and unrelaxed 1-RDM; UCCSD/UCCD | [CC guide](src/gradscf/cc/README.md); [open-shell scope](src/gradscf/cc/OPEN_SHELL.md); in-core molecular implementation |
| GW | Molecular G0W0 with contour deformation, evGW, restricted qsGW and finite-temperature matrix scGW; Gamma/k-point GW | [GW modules](src/gradscf/gw), [Matsubara scGW](src/gradscf/gw/SCGW.md), [periodic GW](src/gradscf/gw/pbc) |
| Integrals and density fitting | Native CPU and JAX reference integrals, packed ERIs, direct J/K, auxiliary-basis RI and full-ERI spectral factorization | [Integral API](src/gradscf/integrals), [compact/direct/DF paths](src/gradscf/integrals/COMPRESSED.md) |

ROHF-based UCCSD uses unrestricted cluster amplitudes on common spatial
orbitals; it is not a separate spin-adapted ROCCSD implementation. Open-shell
triples corrections and explicit UCC Lambda/density facades are not yet provided.

## Differentiation and numerical solvers

GradSCF combines JAX contractions with native integral kernels and explicit
derivative rules. Supported inputs and derivative orders depend on the path:

| Calculation path | Differentiable quantities | Coverage and conditions |
| --- | --- | --- |
| Differentiable SCF | Energies and states versus numerical inputs and model parameters | Implicit or unrolled modes; upstream integral/XC derivatives are required for the selected inputs |
| TDA and CI | Eigenvalues; eigenvectors for coefficient-dependent objectives | Shared isolated-root response; CI coefficient AD requires `gradient_mode="implicit_eigenvector"` |
| Ground-state CC | Energies and amplitudes versus MO integrals | Implicit response at converged roots; fixed topology and orbital ordering; first-order validated contract |
| GW | Quasiparticle roots and selected matrix-scGW responses | Path-specific rules; evGW/qsGW outer self-consistency loops currently have no AD rule |
| Native integrals | Geometry or density response on supported operators/layouts | Read the operator-specific contract; geometry, exponent, and coefficient derivatives are not interchangeable |
| Basis and neural models | Contractions and neural parameters through supported SCF/response paths | Requires derivative support along the complete calculation; fixed-primitive NNAO contraction training is a distinct path |

[gradscf.solvers](src/gradscf/solvers/README.md) owns shared diagonalization,
linear solves, nonlinear iteration, and their derivative rules. Method modules
provide Hamiltonian actions, residuals, and physical convergence conditions.
Eager calculation objects prepare inputs; use functional kernels inside
`jax.jit` and `jax.grad`.

Convergence and response conditioning matter. Check returned residuals,
convergence flags, and validity diagnostics. Derivatives at degeneracies or
unconverged solutions are not generally defined by the isolated-root contracts.
Higher derivatives are supported only on specifically validated paths.

Complete nuclear derivatives also need orbital response and the appropriate
integral/XC derivatives. The generic molecular `mf.nuc_grad_method().kernel()`
entry is currently disabled; it is not the interface for the lower-level
[force-supervision example](examples/train_neural_scf_forces.py).

## Machine learning with electronic structure

**Neural exchange–correlation functionals.**
[model.neural_xc](src/gradscf/model/neural_xc) defines configurable XC models,
features, and molecular bindings. [model.training](src/gradscf/model/training)
provides self-consistent and response-aware training utilities. The
[force-supervision example](examples/train_neural_scf_forces.py) demonstrates a
small neural XC energy with differentiable SCF on a finite test quadrature;
it is a derivative demonstration rather than an accurate production DFT grid.

**Neural atomic-orbital basis sets.**
[model.nnao](src/gradscf/model/nnao/NNAO.md) connects MACE predictions to
normalized Gaussian contractions. The
[basis example](examples/nnao_basis.py) illustrates assembly, and the
[molecular optimization tool](tools/optimize_methane_nnao.py) minimizes converged
RHF energies using direct J/K or DF factors with fixed primitive parameters.
Fixed-geometry optimization results do not establish a transferable basis model;
comparisons must specify the molecule, basis family, and optimization protocol.

Additional neural dispersion models are available in
[model.neural_d](src/gradscf/model/neural_d).

## Documentation and examples

- **CI and CC:** [CI tutorial](examples/ci/restricted_ci.py),
  [restricted CC](examples/cc/restricted_ground.py),
  [open-shell CI/CC](examples/cc/open_shell_ground.py).
- **Periodic calculations:** [HF/DFT and bands](examples/periodic_h2.py).
- **GW:** [finite-temperature scGW](examples/scgw_matsubara_h2.py),
  [implicit scGW response](examples/scgw_implicit_response_h2.py).
- **Numerical development:** [shared solver guide](src/gradscf/solvers/README.md)
  and [operator/solver example](examples/shared_solvers.py).
- **Integrals:** [native build and derivative contracts](src/gradscf/integrals/_native/README.md),
  [compact integrals and density fitting](src/gradscf/integrals/COMPRESSED.md).
- **Migration:** [namespace changes](MIGRATION.md) from GradTDDFT to `gradscf`.

## Validation and development status

Validation includes comparisons with PySCF, independently constructed
Hamiltonians or working expressions, finite differences, and residual and
physical-consistency checks. Numerical tolerances and tested systems are
recorded in the [CC validation report](src/gradscf/cc/VALIDATION.md),
[CI guide](src/gradscf/ci/README.md),
[solver validation report](src/gradscf/solvers/VALIDATION.md), and
[scGW notes](src/gradscf/gw/SCGW.md).

CI and CC currently target in-core reference calculations. Transforming DF
inputs into full MO tensors does not make these implementations DF-CI or DF-CC.
CPU/float64 validation does not establish GPU coverage, and a method's presence
does not establish every reference, property, or derivative combination.

With test dependencies installed and native integrals built, run focused checks:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q tests/solvers tests/ci tests/cc
```

Run `python -m pytest -q` for the full suite. Optional-dependency tests may skip;
inspect the skip reasons with `-ra` when assessing coverage.

## Contributing

Bug reports and focused contributions are welcome through
[GitHub Issues](https://github.com/STOKES-DOT/GradSCF/issues) and pull requests.
For numerical reports, include a minimal example, commit/version, JAX backend,
dtype, molecular geometry and units, basis/XC settings, convergence tolerances,
and the reference result. Add a targeted regression for behavior changes and
keep method equations separate from shared numerical solver implementations.

## Citation

For reproducibility, identify the GradSCF repository and the commit or version
used in your calculation. Cite the theoretical methods and upstream software
relevant to that calculation; method references and implementation attribution
are recorded separately in the [CI references](src/gradscf/ci/REFERENCES.md),
[CC references](src/gradscf/cc/REFERENCES.md), and
[GW module documentation](src/gradscf/gw/__init__.py).

<a id="code-origins"></a>

## License and acknowledgments

Original GradSCF code is distributed under the [MIT license](LICENSE).
The foundational DFT and TDDFT code originates from
[GradTDDFT](https://github.com/STOKES-DOT/GradTDDFT), whose interfaces and modules
have been expanded and reorganized under the `gradscf` namespace.

Adapted and vendored components retain their original notices and licenses:
[PySCF CC contractions](src/gradscf/cc/NOTICE.md),
[native PySCF/libcint sources](src/gradscf/integrals/_native/README.md), and
the [MACE-JAX source record](src/gradscf/model/nnao/UPSTREAM.json).
