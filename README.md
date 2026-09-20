# GradSCF

GradSCF is a JAX toolkit for Hartree-Fock and Kohn-Sham self-consistent-field
calculations, differentiable SCF, response theory, and Neural XC training.
The Python package is `gradscf`; the import namespace
is `gradscf`.

```python
from gradscf import dft, gto, scf, tdscf
from gradscf.model import neural_xc, training
```

GradSCF was previously named GradTDDFT. The canonical imports are now
`gradscf` and `gradscf_tools`; the old import namespaces are no longer shipped.
See [MIGRATION.md](MIGRATION.md) for the name mapping and installation checks.

## Code Origins

The foundational DFT and TDDFT code in GradSCF originates from
[GradTDDFT](https://github.com/STOKES-DOT/GradTDDFT). GradSCF builds on that
codebase with expanded SCF methods, reorganized integral backends, and a
dedicated `gradscf` API. Original copyright notices and licenses are retained.

## Supported Methods

| Category | Methods | Entry points |
|---|---|---|
| SCF (molecules) | RHF, UHF, ROHF, GHF, RKS, UKS, ROKS, GKS; DIIS, damping, level shift, UHF/UKS stability analysis with directed restart | `gradscf.scf`, `gradscf.dft` (PySCF-style facades) |
| Differentiable SCF | `unrolled` and `implicit` adjoint modes (gradients of converged energies/orbitals w.r.t. geometry, basis exponents/contractions, model parameters) | `SCFDifferentiationConfig`, `gradscf.scf.autodiff` |
| Excited states | TDA, full Casida TDHF/TDDFT (restricted and unrestricted), implicit-differentiable Davidson eigensolvers, oscillator strengths and spectra | `gradscf.tdscf`, `gradscf.tddft` |
| GW many-body | Molecular G0W0 with contour deformation (restricted/unrestricted), evGW (Z-update), qsGW (static self-consistent potential), matrix scGW with Galitskii–Migdal energy | `gradscf.gw` (`GW`, `UGW`, `evgw_cd_*`, `qsgw_cd_restricted`, `scgw_cd_restricted`) |
| Periodic GW | Plane-wave product basis, momentum-conserving k-point sampling, q→0 head/wing finite-size corrections; Gamma and k-point KRGW | `gradscf.gw.pbc` (`KRGW`, `g0w0_cd_kpoints`) |
| Periodic SCF | GTH pseudopotentials, Ewald electrostatics, FFT density fitting, Γ/k-point HF and DFT, band structures, q=0 TDA/TDDFT optics | `gradscf.pbc` (`gto`, `scf`, `dft`, `tdscf`, `bands`) |
| Density fitting | Full-ERI spectral factorization (pure JAX, differentiable), native compact RI backend (libcint int3c2e) | `gradscf.df`, `gradscf.integrals` |
| Integrals | Native C++ CPU kernels (vendored libcint/PySCF C sources, no runtime PySCF), Cartesian/spherical, overlap/kinetic/nuclear/dipole/ERI, coordinate JVP/VJP | `gradscf.integrals`, `gradscf.integrals._native` |
| Neural XC | Neural XC functional construction, DM21-style presets, self-consistent and response-aware training | `gradscf.model.neural_xc`, `gradscf.model.training` |
| Neural basis sets (NNAO) | MACE-conditioned per-atom contracted-GTO basis assembly for 34 main-group elements (H–Xe), variational basis training | `gradscf.model.nnao` |
| Dispersion | Neural dispersion corrections | `gradscf.model.neural_d` |

## Differentiable Architecture

Everything below the public facades is a pure-JAX function, so energies,
orbital energies, and spectra are differentiable with respect to geometry,
basis-set parameters, and neural-network weights.

- **SCF adjoints.** `SCFDifferentiationConfig(mode="implicit" | "unrolled")`
  selects between an implicit-function-theorem adjoint (one linear solve at
  the converged point) and direct backpropagation through the SCF loop.
  Geometry gradients match PySCF analytic gradients to ~1e-12 Ha/Bohr, and
  basis exponents/contraction coefficients are differentiable inputs for
  variational basis-set training.
- **Quasiparticle adjoints.** The GW quasiparticle equation is solved
  graphically; its gradients use the implicit function theorem, with the
  quasiparticle renormalization factor `Z = 1/(1 - dΣ/dω)` appearing
  naturally in the denominator. Singular backward rules (satellite regions,
  `Z → 0`) raise explicit errors instead of returning silent fallbacks.
- **Self-consistent stationarity.** qsGW/scGW fixed-point loops are
  differentiated at the stationary point (envelope theorem), so total-energy
  and response gradients do not require unrolling the self-consistent cycles.
- **Finite-temperature scGW responses.** `scgw_matsubara_restricted` exposes
  implicit responses of the Matsubara-grid scGW model to density-fitting
  vertex scalings (`examples/scgw_implicit_response_h2.py`).

## Neural XC and Neural Basis Sets

**`gradscf.model.neural_xc`** builds neural exchange-correlation functionals
that run inside the standard SCF/TDDFT machinery (restricted/unrestricted,
ground and response). Functionals bind to molecules
(`bind_to_molecule_for_scf`), can mix exact exchange and nonlocal features,
and train through self-consistent loops with `gradscf.model.training`
(energy, force, excitation, and density targets; checkpoints included).

**`gradscf.model.nnao`** produces neural-network atomic-orbital (NNAO) basis
sets: a MACE graph model predicts signed contraction coefficients per shell
(s/p/d output heads), which are assembled into Gaussian-normalized contracted
GTO basis sets for 34 main-group elements. Bases are optimized variationally
by directly minimizing converged SCF energies (`tools/optimize_methane_nnao.py`),
so the resulting sets beat same-size Pople-style bases at a fraction of the
primitives of correlation-consistent sets.

## Usage Examples

### Molecular SCF and TDDFT

```python
from gradscf import dft, gto, tdscf

mol = gto.M(atom="O 0 0 0.117; H 0 0.755 -0.471; H 0 -0.755 -0.471",
            basis="def2-svp")
mf = dft.RKS(mol, xc="pbe").run()          # UKS / ROKS / GKS; HF = xc="hf"
td = tdscf.TDDFT(mf, nstates=5)
e, xy = td.kernel()                        # excitation energies (Ha), amplitudes
```

### Molecular GW

```python
from gradscf import dft, gto
from gradscf.gw import GW, UGW, evgw_cd_restricted, qsgw_cd_restricted

mol = gto.M(atom="H 0 0 0; F 0 0 0.92", basis="cc-pvdz")
mf = dft.RKS(mol, xc="hf").run()

gw = GW(mf, nw=100).run()                  # G0W0, contour deformation
print(gw.mo_energy)                        # quasiparticle energies (Ha)
```

### Periodic SCF and k-point GW

```python
import numpy as np
from gradscf.pbc import gto, scf
from gradscf.gw.pbc import KRGW

cell = gto.M(atom="H 3.0 3.0 2.3; H 3.0 3.0 3.7", a=np.eye(3) * 6.0,
             unit="Bohr", basis="gth-szv", pseudo="gth-pade",
             mesh=(17, 17, 17))
mf = scf.KRHF(cell, kpts=cell.make_kpts([1, 1, 1])).run()
gw = KRGW(mf, nw=60).run()                 # q->0 head/wing included by default
```

### Neural XC functional

```python
from gradscf import gto
from gradscf.model import neural_xc

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
functional = neural_xc.Functional(model=..., name="my_xc")  # see neural_xc docs
mf_like = neural_xc.make_neural_xc_functional(functional)
# train with gradscf.model.training.MolecularTrainingConfig / Trainer
```

### Neural basis set (NNAO)

```python
from gradscf.model.nnao import prepare_direct_basis, supported_elements

basis = prepare_direct_basis("C 0 0 0; H 0 0 1.09; H 0.51 0 0.94; ...")
print(supported_elements())                # 34 main-group elements (H–Xe)
```

### Differentiable SCF gradient

```python
from gradscf import dft, gto
from gradscf.scf import SCFDifferentiationConfig

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
mf = dft.RKS(mol, xc="hf").run()
grad = mf.nuc_grad_method().kernel()     # analytic nuclear gradient
# implicit/unrolled adjoint modes are selected via SCFDifferentiationConfig
# in the lower-level solvers (see tests/pbc/test_gradients.py)
```

### scGW (finite-temperature, implicit response)

```python
# examples/scgw_matsubara_h2.py        — finite-T scGW on the Matsubara grid
# examples/scgw_implicit_response_h2.py — dE/ds for DF-vertex scaling
from gradscf.gw import scgw_matsubara_restricted
```

## Installation

```bash
python -m pip install -e ".[dev,upstreams]"        # tests, PySCF, jax-xc
python -m pip install -e ".[dev,reproducibility]"  # manuscript evaluation stack
python -m pip install -e ".[nnao]"                 # MACE-jax for NNAO models
PYTHONPATH=src python -m gradscf.integrals._native.build  # native CPU integrals
```

Tests run on CPU with float64:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q
```

## Documentation Map

- [MIGRATION.md](MIGRATION.md) — old→new namespace mapping (`td_graddft`,
  `gradscf.neural_xc`, `gradscf.xc_backend`, top-level `nnao`, helper modules)
- `src/gradscf/gw/` — GW module docstrings with literature references
  (Hedin 1965; Hybertsen–Louie 1986; Godby–Needs 1989; Ren 2012;
  Faleev/van Schilfgaarde/Kotani qsGW; PRB 83, 245122 q→0 corrections)
- `src/gradscf/gw/SCGW.md` — scGW design notes
- `src/gradscf/integrals/COMPRESSED.md` — native compact-integral backend
- `src/gradscf/model/nnao/NNAO.md` — NNAO usage and scope

## License

MIT (see LICENSE). Vendored upstream sources keep their original licenses
and notices.
