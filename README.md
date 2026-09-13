# GradSCF

GradSCF is a JAX toolkit for Hartree-Fock and Kohn-Sham self-consistent-field
calculations, differentiable SCF, response theory, and Neural XC training.
The Python package is `gradscf`; the import namespace
is `gradscf`.

```python
from gradscf import dft, gto, scf, neural_xc, tdscf, training
```

GradSCF was previously named GradTDDFT. The canonical imports are now
`gradscf` and `gradscf_tools`; the old import namespaces are no longer shipped.
See [MIGRATION.md](MIGRATION.md) for the name mapping and installation checks.

## Code Origins

The foundational DFT and TDDFT code in GradSCF originates from
[GradTDDFT](https://github.com/STOKES-DOT/GradTDDFT). GradSCF builds on that
codebase with expanded SCF methods, reorganized integral backends, and a
dedicated `gradscf` API. Original copyright notices and licenses are retained.

## Inherited v1.0.0 Scope

The first release contains:

- restricted and unrestricted JAX SCF;
- differentiable `unrolled` and `implicit` SCF modes (`expl` / `impl` aliases remain supported);
- matrix-free TDA and full Casida TDDFT with Davidson solvers;
- strict conventional XC response through `jax-xc`;
- residual Neural XC models with semilocal, fixed-cache HFX, and optional
  fixed-cache PT2 channels;
- the training and evaluation drivers used in the GradTDDFT manuscript;
- selected checkpoints, final inference tables, references, and figures under
  `reproducibility/v1.0.0/`.

Raw datasets, HDF5 integral caches, profiling runs, temporary remote scripts,
and unrelated research drivers are not part of the release.

## Installation

Create an environment with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,upstreams]"
```

The `upstreams` extra installs `jax-xc`. Independent PySCF reference tests
use the separate `comparison-tests` extra. For manuscript scripts and
checkpoint evaluation, install:

```bash
python -m pip install -e ".[dev,reproducibility]"
```

GPU SCF and response runs require a CUDA-enabled JAX build. Molecular
integrals use the private native CPU library. Confirm the active JAX backend
before a long run:

```bash
python - <<'PY'
import jax
from gradscf.xc_backend import jax_xc_backend_info

print(jax.devices())
print(jax_xc_backend_info())
PY
```

Scientific reference calculations should enable JAX float64 before arrays are
constructed:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Integral backends and trainable basis parameters

`gradscf.integrals` is the canonical integral namespace. Existing matrix APIs
such as `overlap_matrix`, `build_hcore`, and `eri_tensor` retain their numerical
behavior through the JAX reference backend. Integral-input assembly now lives
under `gradscf.integrals.assembly`. Basis loading and all retained basis assets live
in `gradscf.integrals.basis_data`; numerical grids and AO grid evaluation live in
`gradscf.integrals.grids`. Removed legacy import paths are listed in `MIGRATION.md`.

The private CPU backend vendors pinned libcint/PySCF C sources and builds
offline without importing PySCF at runtime:

```sh
PYTHONPATH=src python -m gradscf.integrals._native.build
```

Enable JAX float64 before constructing parameters:

```python
import jax
jax.config.update("jax_enable_x64", True)
from gradscf import integrals

topology, parameters = integrals.prepare_basis(
    atom="H 0 0 0; H 0 0 .74", basis="sto-3g",
)
plan = integrals.make_plan(topology, backend="native")
overlap = plan.evaluate("overlap", parameters)
eri = plan.evaluate("eri", parameters)
```

`BasisTopology` holds static shell structure. `BasisParameters` is a JAX pytree
containing raw exponents, raw contraction coefficients, basis centers, and
nuclear coordinates. `IntegralPlan` caches only topology and index layouts;
normalization and all parameter values are rebound on every call. Basis centers
and nuclei may move independently. Dipole plans default to the nuclear-charge
center, matching the reference backend; pass `origin=` to choose another origin.

Native evaluation supports CPU float64, Cartesian/spherical S/T/V/dipole/full
ERI, and `jax.jit`. Integral plans support first-order coordinate JVP/VJP
through native derivative kernels, including independent nuclear/basis centers
and dipole origins. No PySCF runtime or bridge is used by this path. Exponent
and contraction-coefficient derivatives still require `backend="jax_reference"`;
unsupported native basis or higher derivatives raise rather than returning zero. Inspect
`integrals.backend_capabilities(...)` for backend-specific contracts.

SCF `integral_backend="native"` (also accepted as `"cpu"`/`"libcint"`) uses
the private native library. No external chemistry runtime is required. See the
[native backend guide](src/gradscf/integrals/_native/README.md)
for pinned sources, build requirements, ABI, and supported operators.

Integral implementations and build resources are grouped under
`src/gradscf/integrals/`: public APIs and assembly at the package root,
backend adapters in `backends/`, and the private loader, C/C++ sources,
vendored dependencies, and CMake files in `_native/`. Build with
`python -m gradscf.integrals._native.build`; the old `_native` alias is removed.

## Unrestricted Hartree-Fock

Use `scf.UHF` for independent alpha/beta orbitals with full exact exchange:

```python
import jax
jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf

mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", unit="Angstrom",
    charge=1, spin=1,  # spin = N_alpha - N_beta
)
mf = scf.UHF(mol, integral_backend="cpu", execution_device="cpu").run()
if not mf.converged:
    raise RuntimeError("UHF did not converge")
print("UHF energy / Ha:", mf.e_tot)
dm_alpha, dm_beta = mf.make_rdm1()
```

The facade exposes `.kernel()`, `.run()`, `.e_tot`, `.converged`, and spin-stacked
`.mo_coeff`, `.mo_energy`, and `.mo_occ`. Its XC is fixed to HF; use `dft.UKS`
for DFT. It reuses the existing UKS molecular-input/reference pipeline, including
grid preparation; its density-fitting, direct-SCF, and explicit nuclear-gradient
methods are currently unsupported.

For grid-free ground-state calculations from real AO integrals, use
`scf.run_uhf_from_integrals(overlap=..., hcore=..., eri=..., nalpha=...,
nbeta=..., nuclear_repulsion=..., config=scf.UHFConfig(...))`.
`scf.run_uhf(basis=..., nalpha=..., nbeta=...)` constructs Cartesian integrals
with JAX. Both return `scf.UHFResult` with separate alpha/beta fields and accept
`init_density_alpha`/`init_density_beta` for a spin-broken initial guess.

## Restricted Open-Shell HF and KS

`scf.ROHF` and `dft.ROKS` optimize one common set of spatial orbitals with
double, single, and zero occupations. They use a Roothaan effective Fock
constructed from separate alpha/beta potentials. ROKS here means the high-spin,
restricted open-shell KS convention used by PySCF, not multiplet-sum ROKS for
excited singlets.

```python
import jax
jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, scf

mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", unit="Angstrom",
    charge=1, spin=1,
)
hf = scf.ROHF(mol, execution_device="cpu").run()
ks = dft.ROKS(mol, xc="pbe", grids_level=0, execution_device="cpu").run()
assert hf.converged and ks.converged
print(hf.e_tot, ks.e_tot)  # Hartree
print(hf.mo_occ)          # 0/1/2 occupations; mo_coeff has shape (nao, nao)
dm_alpha, dm_beta = hf.make_rdm1()
```

ROKS supports the existing LDA/GGA and global-hybrid XC paths, including PBE
and B3LYP, through `jax-xc`. `scf.run_roks_from_integrals` accepts AO values,
first derivatives, and grid weights alongside the integrals. For grid-free
ROHF, use `scf.run_rohf_from_integrals` or `scf.run_rohf` with a Cartesian basis.
The low-level solvers accept explicit `nalpha`/`nbeta` counts and return common
orbitals plus separate spin densities and Fock matrices. Convergence requires
energy, density-change, and orbital-gradient tolerances simultaneously.

This implementation covers real-orbital, full-integral ground states, including
the closed-shell limit and either sign of spin polarization. Density fitting,
direct SCF, meta-GGA, explicit nuclear gradients, and RO-reference TDA/TDDFT
are not implemented. The molecular facades reuse UKS integral/grid preparation;
they do not run an unrestricted ground-state solver.

## Generalized HF and KS

`scf.GHF` and `dft.GKS` use complex spinor orbitals in alpha-then-beta AO order.
The coefficient and density matrices have shape `(2*nao, 2*nao)`, and each
occupied spinor holds one electron. Only the total electron number is fixed;
`mol.spin` guides the initial guess, rather than constraining the final spin.

```python
import jax
jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, scf

mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", charge=1, spin=1)
hf = scf.GHF(mol, execution_device="cpu").run()
ks = dft.GKS(mol, xc="pbe", collinear="col", execution_device="cpu").run()
assert hf.converged and ks.converged
dm_spinor = hf.make_rdm1()
```

Pass a full complex Hermitian initial density to `.kernel(dm0=...)` or
`.run(dm0=...)` to start from mixed alpha/beta spinors. Off-diagonal exchange
blocks and the imaginary parts of orbitals and DIIS errors are retained.

GKS exposes two XC schemes matching the corresponding PySCF conventions:

- `collinear="col"` (default): fixed-axis spin densities with LDA/GGA and
  supported global hybrids such as PBE and B3LYP. Semilocal XC uses the alpha
  and beta diagonal density blocks; this approximation is not generally
  invariant under global spin rotations.
- `collinear="ncol"`: noncollinear LDA using the local total density and
  magnetization magnitude. For example, use `xc="lda_x"`; the XC potential
  includes all three magnetization components and the zero-magnetization limit.

`scf.run_ghf_from_integrals` is grid-free; `scf.run_ghf` builds Cartesian JAX
integrals. `scf.run_gks_from_integrals` additionally accepts real spatial AO
values, their first derivatives, and grid weights. Both integral interfaces
take a spatial overlap matrix and full real spatial ERIs; `hcore` may be either
a spatial matrix or a complex Hermitian spinor matrix with explicit spin-mixing
terms. The molecular facade does not construct SOC or relativistic integrals.

Multi-collinear (`mcol`) XC, noncollinear GGA, meta-GGA, density fitting, direct
SCF, generalized TDA/TDDFT, and explicit nuclear gradients are not implemented.
Validation uses CPU float64/complex128; GPU execution and differentiation at
degenerate spinor eigenvalues have not been validated.

## Conventional DFT, TDA, and full TDDFT

The public facade follows the PySCF workflow. This example runs a conventional
B3LYP calculation entirely through the GradSCF API:

```python
import jax
jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto

mol = gto.M(
    atom="""
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """,
    basis="def2-svp",
    unit="Angstrom",
    spin=0,
    charge=0,
)

mf = dft.RKS(
    mol,
    xc="b3lyp",
    grids_level=0,
    integral_backend="cpu",
    execution_device="cpu",
)
energy_h = mf.kernel()
if not mf.converged:
    raise RuntimeError("RKS did not converge")
print("E0 / Ha:", energy_h)

tda = mf.TDA(nstates=3)
tda_result = tda.kernel()
print("TDA / eV:", tda.e_ev)
print("TDA oscillator strengths:", tda.oscillator_strength())

full = mf.TDDFT(nstates=3)
full_result = full.kernel()
if not bool(full_result.converged):
    raise RuntimeError("full-TDDFT Davidson did not converge")
print("full-TDDFT / eV:", full.e_ev)
```

Use `integral_backend="native"` for molecular CPU integrals and
`execution_device="gpu"` for SCF/response in a CUDA-enabled JAX environment.
The resulting fixed molecular integrals are reused by SCF and response calls.

A complete PySCF-versus-GradSCF B3LYP comparison is provided in
`tests/comparisons/compare_pyscf_vs_jax_tddft_no_neural.py`.

## Build a Neural XC Functional

The manuscript model predicts local mixing coefficients for the B3LYP
semilocal decomposition and optional HFX/PT2 channels:

```text
e_xc^NN(r) = sum_k c_k(r) e_k^semilocal(r)
             + c_HF(r) e_HF(r)
             + c_PT2(r) e_PT2(r)       # optional
```

Construct the model through the public `neural_xc` namespace:

```python
from gradscf import neural_xc
from gradscf.xc_backend import b3lyp_component_basis

functional = neural_xc.Functional(
    architecture="graddft_residual",
    hidden_dims=(128, 128, 128, 128),
    semilocal_xc=b3lyp_component_basis(),
    input_feature_mode="canonical",
    include_hfx_channel=True,
    ground_state_hf_mode="nograd",
    include_pt2_channel=False,
    ground_state_pt2_mode="off",
    response_hf_mode="approx",
    response_pt2_mode="approx",
    name="my_neural_xc",
)
```

The channel order is fixed:

```text
[semilocal_1, ..., semilocal_n, pt2?, hf]
```

In v1.0.0, ground-state nonlocal channels support only `off` and `nograd`:

- HFX `nograd` requires a fixed `hfx_fxx` cache.
- PT2 `nograd` requires fixed `pt2_local`; self-consistent Fock construction
  additionally requires `pt2_fock_response`.
- Density-updated ground-state HFX/PT2 `scf` modes are not public in v1.0.0.

Build these caches once when the molecular reference is prepared. They are not
recomputed after each parameter update:

```python
from gradscf.scf import restricted_molecule_from_spec_with_jax_rks

reference = restricted_molecule_from_spec_with_jax_rks(
    atom="H 0 0 0; H 0 0 0.74",
    basis="def2-svp",
    xc_spec="b3lyp",
    integral_backend="native",
    init_guess="hcore",
    grids_level=2,
    compute_local_hfx_features=True,
    compute_local_hfx_aux=False,
    compute_local_pt2_features=False,
)
```

Set `compute_local_pt2_features=True`, `include_pt2_channel=True`, and
`ground_state_pt2_mode="nograd"` to train with the fixed PT2 channel.

## Train the Neural XC Model

One API handles fixed-density, explicit-SCF, and implicit-SCF objectives. Target
names include units and physical meaning: `target_e0_total_h` is a ground-state
total energy in hartree; `target_s1_total_h` is an S1 total energy; and
`target_excitation_gaps_h` contains excitation gaps.

```python
import jax
import jax.numpy as jnp
import optax

from gradscf import training

datum = training.MolecularTrainingDatum(
    molecule=reference,
    target_e0_total_h=jnp.asarray(-1.1372838345),
)
config = training.MolecularTrainingConfig(
    mode="self_consistent",
    scf_gradient_mode="impl",       # use "expl" for unrolled SCF
    e0_total_mse_weight=1.0,
    e0_total_mae_weight=1.0,
    scf_max_cycle=32,
    scf_convergence_metric="energy",
    scf_conv_tol_energy=1e-8,
)

state = training.create_train_state_from_molecule(
    functional,
    jax.random.PRNGKey(0),
    reference,
    optax.adam(1e-3),
)
train_step = training.make_molecular_train_step(
    functional,
    training_config=config,
)

for step in range(100):
    state, metrics = train_step(state, (datum,))
    print(step, float(metrics["total_loss"]))
```

For excited-state training, activate explicit loss components rather than an
ambiguous generic S1 weight:

```python
excited_config = training.MolecularTrainingConfig(
    mode="self_consistent",
    scf_gradient_mode="impl",
    excited_state_solver="tda",
    excitation_gap_mse_weight=1.0,
    excitation_gap_mae_weight=1.0,
    excitation_gap_nstates=1,
)
excited_datum = training.MolecularTrainingDatum(
    molecule=reference,
    target_excitation_gaps_h=jnp.asarray([0.40]),
)
```

The full paper-scale H2 example is
`tests/comparisons/h2_fci_self_consistent_train.py`. Production H2, H2+, N2, QM9, and
QM9GWBSE commands are listed below.

## Save, Restore, and Infer

Save model parameters together with the architecture and channel configuration:

```python
checkpoint = "outputs/my_neural_xc.msgpack"
training.save_params_checkpoint(
    checkpoint,
    state.params,
    metadata={
        "architecture": "graddft_residual",
        "hidden_dims": [128, 128, 128, 128],
        "ground_state_hf_mode": "nograd",
        "ground_state_pt2_mode": "off",
    },
)

params = training.load_params_checkpoint(checkpoint, template=state.params)
```

Ground-state inference must use the same fixed-density or self-consistent policy
as training:

```python
energy_h = training.predict_ground_state_total_energy(
    params,
    functional,
    reference,
    training_config=config,
)
converged_molecule = training.predict_ground_state_molecule(
    params,
    functional,
    reference,
    training_config=config,
)
```

Run TDA or full-TDDFT from the converged molecule:

```python
from gradscf.spectra import HARTREE_TO_EV

gaps_h = training.predict_excitation_energies(
    params,
    functional,
    converged_molecule,
    nstates=3,
    use_tda=True,
)
strengths = training.predict_oscillator_strengths(
    params,
    functional,
    converged_molecule,
    nstates=3,
    use_tda=True,
)
print("gaps / eV:", gaps_h * HARTREE_TO_EV)
print("oscillator strengths:", strengths)
```

Set `use_tda=False` for full Casida TDDFT. Excitation-energy derivatives use a
converged-vector implicit differential and do not backpropagate through the
Davidson iteration history. TDA also provides opt-in implicit eigenvector
gradients for oscillator-strength objectives.

## Manuscript Workflows

Only manuscript-facing drivers are included in `tools/`:

| Manuscript task | Entry point |
| --- | --- |
| Conventional PySCF validation | `tests/comparisons/compare_pyscf_vs_jax_same_xc.py` |
| Benzene Davidson convergence | `tests/comparisons/trace_benzene_pbe_tddft_davidson.py` |
| H2+ ground-state dissociation | `tests/comparisons/h2plus_fci_ground_train5_dense100.py` |
| H2 ground-state dissociation | `tests/comparisons/h2_self_consistent_ground_train5_dense100_vs_fci.py` |
| N2 ground-state dissociation | `tests/comparisons/n2_ccsdt_ground_train5.py` |
| H2/N2 S1 TDA dissociation | `tests/comparisons/h2_s1_tda_train5_dense100_vs_fci.py` |
| QM9 ground and QM9GWBSE S1 training | `tests/comparisons/closed_shell_s1_self_consistent_train.py` |
| Released checkpoint inference | `tests/comparisons/evaluate_closed_shell_checkpoint.py` |
| QM9/QM9GWBSE baselines and figures | `tests/comparisons/compute_qm9_ground_classic_baselines.py`, `tests/comparisons/compare_qm9_pyscf_vs_jax_tda.py`, `tools/plot_qm9_reference_structures.py`, `tools/plot_qm9_val_bars_with_structures.py` |

Example paper commands:

```bash
python tests/comparisons/h2_self_consistent_ground_train5_dense100_vs_fci.py \
  --basis def2-tzvp \
  --grids-level 2 \
  --ground-state-hf-mode nograd \
  --ground-state-pt2-mode off \
  --steps 2000

python tests/comparisons/h2_s1_tda_train5_dense100_vs_fci.py \
  --basis def2-tzvp \
  --grids-level 2 \
  --include-pt2-channel \
  --response-pt2-mode strict \
  --steps 2000
```

Selected checkpoints, compact reference CSVs, final inference results, figures,
and provenance are documented in
[`reproducibility/v1.0.0/README.md`](reproducibility/v1.0.0/README.md). Verify
the artifact manifest from that directory with:

```bash
shasum -a 256 -c SHA256SUMS
```

## Traditional XC Support

Conventional XC labels are parsed by `gradscf.xc_backend.jax_libxc` and
evaluated with `jax-xc`. Strict default components include:

```text
lda_x, lda_c_pw, lda_c_vwn, lda_c_vwn_rpa
gga_x_b88, gga_x_pbe, gga_x_wpbeh
gga_c_lyp, gga_c_pbe
```

Common composites include `lda`, `svwn`, `pbe`, `pbe0`, `b3lyp`, BHandHLYP,
HSE03, and HSE06. B3LYP is resolved as:

```text
0.20*hf + 0.08*lda_x + 0.72*gga_x_b88
        + 0.19*lda_c_vwn_rpa + 0.81*gga_c_lyp
```

Installed functionals outside the validated set require
`allow_experimental_jax_xc=True`.

## Repository Layout

```text
src/gradscf/       DFT, SCF, TDDFT, Neural XC, training, and data APIs
src/gradscf_tools/ Small supporting analysis utilities
examples/             Two runnable manuscript-oriented examples
tools/                Manuscript training, validation, and plotting drivers
tests/                Focused unit and regression tests
reproducibility/      Versioned checkpoints, results, references, and figures
```

## Testing

Run the focused release checks:

```bash
pytest -q tests/test_neural_xc_public_api.py
pytest -q tests/test_tddft_eigensolvers.py
pytest -q tests/test_molecular_training_api.py
pytest -q tests/test_workflows_config.py
```

Some reference comparisons require PySCF or generated input data; GPU
validation requires CUDA-enabled JAX. Large integral/basis sweeps are not part of the compact release gate;
the manuscript dissociation and QM9 artifacts provide the corresponding
end-to-end evidence.

## v1.0.0 Limitations

- Ground-state HFX/PT2 supports `off` and fixed-cache `nograd`; density-updated
  HFX/PT2 `scf` modes are not released.
- Neural local-HF strict response remains fail-fast; released Neural XC
  checkpoints use the validated approximate HFX response path.
- PT2 strict response is a post-hoc CIS(D)-type correction; SCS/SOS scaling is
  not included.
- Full-TDDFT excitation energies are differentiable through the converged
  symplectic Rayleigh quotient. Full-TDDFT X/Y eigenvector gradients are not
  exposed in v1.0.0.
- Geometry optimization and analytical nuclear gradients are outside the
  v1.0.0 release scope.

## License and Upstreams

GradSCF is released under the MIT License. It interoperates with JAX, Flax,
Optax, and `jax-xc`; PySCF is an optional independent test reference.
Third-party data and source snapshots retain their original licenses and notices.

## Standalone runtime and optional comparisons

PySCF and gpu4pyscf are not used by production GradSCF modules or computational
tools. Build the private CPU library with
`PYTHONPATH=src python -m gradscf.integrals._native.build`. JAX handles SCF, XC
composition, grid evaluation and parameter transforms.

Independent PySCF checks are confined to `tests/`; install
`pip install -e ".[comparison-tests]"` only when running those comparisons.
Their executable entry points live in `tests/comparisons/`. Native experiment
CLIs in `tools/` retain finite-difference/invariance checks without PySCF.

Default initialization is `hcore` (or `1e`); explicit density matrices are also
supported. External MINAO/atomic/SAP/checkpoint guesses and the external GPU
integral backend are removed and fail explicitly. Native `jk_backend="df"`
factorizes native full AO ERIs; it is not an auxiliary-basis three-center DF
calculation and does not avoid allocating full integrals. Native direct J/K
remains unsupported; the explicit JAX reference direct mode remains available.

All basis resources are preserved. Upstream Python-format basis files are
packaged as inert `.pydata`; GradSCF reads literal data and aliases without
executing upstream imports or parsing code. ECP/spinor resource availability
does not imply support for those operators in the native backend.

## Difficult open-shell SCF solutions

UKS/UHF `energy_and_residual` convergence requires energy change, density RMS
and the raw-Fock occupied-virtual gradient to pass. `conv_tol_grad` is exposed
on UKS/UHF configurations and facades. DIIS uses relative Gram regularization;
unrestricted level shifting uses the single-spin projector `S - S D_sigma S`.
Removing a reporting shift does not refill occupations or silently change roots.

For explicit multi-start calculations,
`gradscf.scf.init_guess.orbital_rotation_guesses(C, amplitudes=(0., .1), seed=...)`
generates metric-preserving rotations of a complete orbital matrix. Run each
guess, reject unconverged candidates, and compare energies. This does not prove
that the lowest sampled stationary state is the global ground state.

The `gradscf.scf.orbital_optimization` module provides
`minimize_uks_from_integrals`, `minimize_roks_from_integrals`, and
`minimize_gks_from_integrals`, also exported from `gradscf.scf`. These use JAX
and the existing Optax dependency; no separate SciPy optimizer extra is needed.
They optimize fixed occupations with the unitary Cayley update
`C = C0 (I - kappa/2)^(-1) (I + kappa/2)` for anti-Hermitian `kappa`, and support the existing
HF/LDA/GGA/global-hybrid energy implementations. Results are array-valued PyTrees
usable under JIT; `stationary` tests the true final tangent-gradient norm.

## SCF backward modes

One `SCFDifferentiationConfig` controls both the existing differentiable DFT
solver and the orbital solver. The canonical modes are `implicit` and
`unrolled`; historical `impl` and `expl` remain accepted aliases.

```python
from gradscf.scf import SCFDifferentiationConfig, DifferentiableSCFConfig

backward = SCFDifferentiationConfig(
    mode="implicit", tolerance=1e-9, max_iter=20, require_converged=True,
)
dft_config = DifferentiableSCFConfig(
    mode="self_consistent", differentiation=backward,
)
# With prepared arrays and a fixed integer occupation topology:
# result = minimize_uks_from_integrals(**inputs, differentiation=backward)
# Switch only backward.mode to "unrolled" to differentiate the same forward solve.
```

- `implicit` differentiates the selected stationary branch through its residual
  equation. It reuses the SCF GMRES adjoint solver and checks the actual linear
  residual. An unconverged forward state or failed adjoint produces nonfinite
  backward values instead of silently returning an approximate or zero gradient.
  Forward diagnostics remain available. Nonzero adjoint `regularization`
  explicitly changes the response; the default is zero.
- `unrolled` differentiates the actual finite sequence of JAX updates, including
  initial-orbital dependence. It can differentiate a truncated solve; its gradient
  need not agree with a stationary implicit derivative before response convergence.
  The fixed scans use checkpointing and a discrete backtracking search. This mode
  generally uses more memory and differentiation work.

Both orbital modes use identical forward iterations. Bounded residual corrections
address soft modes; up to three differentiable local refinements also run at a converged
state so a symmetry-preserving primal does not freeze an unfinished tangent
response. Corrections must reduce the true gradient; refinements must keep it
within the requested tolerance. All accepted steps obey a recorded energy-roundoff
allowance and rotation bound. No automatic root selection or global stability
claim is made; derivatives are local to a branch and piecewise iteration decisions.
Spaces with at most 32 orbital coordinates use a directly assembled Hessian for
these local corrections to keep higher-order JAX compilation manageable; larger
spaces use matrix-free Hessian-vector products.
The Cayley map has the same tangent at zero as an exponential rotation, while
its linear-solve derivatives keep force-training compilation manageable.

Orbital `mo_occ` is a static 0/1 topology: close over a NumPy array/tuple before
JIT. Continuous inputs (integrals, overlap, AO/grid arrays, initial coefficients)
remain differentiable. `orthonormalize_initial=True` transports a fixed seed
when the overlap changes; Cholesky metric derivatives retain the AO constraint
response. Native coordinate derivatives are supported; native contraction/exponent
integral derivatives remain unsupported and raise explicitly. Their SCF composition
is tested using the explicitly selected JAX reference integral backend.

An explicit `DifferentiableSCFConfig.differentiation` object takes precedence over
legacy backward fields. Without it, existing DFT defaults are preserved, including
`require_converged_iterates=False`; use the shared policy for strict implicit
convergence enforcement. Both `params` and differentiable `fixed_point_args`
receive implicit derivatives. The default implicit rule uses a differentiable
root and checked linear solve, so solution and adjoint dependence are retained
under nested AD. JVP, mixed second derivatives, and an analytic scalar third
derivative are regression-tested. Custom optimized VJP hooks remain available;
higher-order use requires the hooks themselves to be differentiable, and direct
forward mode is provided by the default root path.
Training workflows honor `objective.scf_gradient_mode`. Their existing
`recover_nonfinite_steps` option can retry a failed unrolled training step using
implicit backward; disable this recovery option when the mode must remain fixed.

Implementation boundaries are `scf/autodiff.py` (policy and residual interface),
`scf/implicit.py` (adjoint algebra), `scf/_orbital_solver.py` (JAX iterations),
`scf/orbital_optimization.py` (orbital constraints and integral inputs), and
`scf/differentiable.py` (existing functional/molecule adapters). Full auxiliary-DF,
MGGA, and generalized excited-state response support are not implied.
Reproducible comparisons live in `tests/comparisons/multistart_open_shell.py`
and `orbital_fallback_matrix.py`.

## Force supervision

The force API accepts a scalar `energy_fn(params, coordinates)` with explicit
continuous inputs. Coordinates are Bohr; Hartree energies give Hartree/Bohr forces.

```python
import jax
from gradscf.training import energy_and_forces, make_force_loss_and_grad

# energy_fn closes over the chosen SCF method and differentiation policy.
prediction = energy_and_forces(energy_fn, params, coordinates)
loss_and_grad = jax.jit(make_force_loss_and_grad(energy_fn))
loss, params_grad = loss_and_grad(params, coordinates, target_forces)
```

`force_matching_loss` is `0.5 * mean((forces - target_forces)**2)` over all
atoms and Cartesian components. It retains the mixed parameter/coordinate
derivative, which can be passed directly to an Optax update. The energy callback
owns SCF convergence and must not detach quantities needed for the desired response.

Native coordinate JVP/VJP operators are differentiable in their direction and
cotangent at fixed geometry/basis. This supports force-loss gradients for model
parameters that change the energy functional. Pure second-coordinate native
integrals and native basis-parameter derivatives still raise explicitly; a
complete nuclear Hessian or force training of a native basis predictor is not
implied. The SCF eigensolver retains the existing near-degeneracy regularization;
the high-order SCF validations use continuous, nondegenerate solution branches.
Overlap inverse-square-root derivatives use a Sylvester solve and remain smooth
at repeated positive overlap eigenvalues, including second derivatives. When
overlap eigenvalue clipping is active, the previous regularized response is
retained; exact smooth-response guarantees apply above the clipping cutoff.

`examples/train_neural_scf_forces.py --mode implicit` (or `unrolled`) runs a
complete H2/3-21g native-integral/tanh-network SCF regression, checks force and
force-loss derivatives by finite differences, and performs an Adam update.
Its small co-moving quadrature is a derivative test, not a production XC model.

## Periodic HF, DFT and response

`gradscf.pbc` provides neutral three-dimensional periodic Gaussian calculations.
The first backend uses analytic Cartesian Gaussian Fourier coefficients and
JAX FFT density fitting; it does not call PySCF or the molecular integral
bridge. GTH data and provenance are packaged under `integrals/periodic/`.

```python
import jax
import numpy as np
from gradscf.pbc import gto, dft, tdscf

jax.config.update("jax_enable_x64", True)
cell = gto.M(
    atom="H .2 .3 .4; H 1.6 .3 .4",
    a=np.eye(3) * 6, unit="Bohr",
    basis="gth-szv", pseudo="gth-pade", mesh=(41, 41, 41),
)
mf = dft.KRKS(cell, kpts=cell.make_kpts((2, 1, 1)), xc="pbe").run()
td = tdscf.TDDFT(mf, nstates=1).run()
print(mf.e_tot, td.e)  # Hartree per cell; excitation energies in Hartree
```

Ground-state classes are `RHF/UHF/RKS/UKS` at Gamma and
`KRHF/KUHF/KRKS/KUKS` for complete uniform k meshes. LDA, GGA and global
hybrids use the existing XC implementation. Occupied band counts are fixed
at each k point: metals, smearing, charged cells and lower-dimensional
electrostatics are not supported. Lattice vectors are rows; k vectors are
Cartesian radians/Bohr. `Cell.spin` is Nalpha-Nbeta **per primitive cell**,
independent of the k mesh. For PySCF KUHF/KUKS comparisons set reference `nelec`
to `(nk*nalpha, nk*nbeta)` explicitly. Internal and returned matrices retain their k axis,
including Gamma: restricted density `(nk,nao,nao)`, unrestricted density
`(2,nk,nao,nao)`. Call `cell.build()` after changing cell specifications.

Named initial assets include `gth-szv/dzvp/tzvp` and
`gth-pade/pbe/blyp`; explicit GTH parameter and basis dictionaries are accepted.
Existing molecular basis data remain available. This is a pseudopotential
implementation, not an all-electron periodic backend. Use odd FFT meshes;
converge `mesh` explicitly. `precision` controls Ewald truncation and does not
certify FFT accuracy. The `exxdiv` options are `'ewald'` and `None`; comparisons
must use the same choice, mesh, pseudopotential, basis and k sampling.

`TDA`, `TDHF` and `TDDFT` implement spin-conserving q=0 response. TDA uses
operator Davidson, and full Gamma response reuses the real TDHF solver.
Complex k-mesh full response currently uses a bounded dense solve, limited to
256 particle-hole transitions by default (`max_dense`). Excitation energies
and X/Y amplitudes are available. Restricted Gamma references also expose
`td.oscillator_strength(gauge="velocity")`, including the nonlocal GTH velocity
correction. `pbc.optics.broaden_spectrum(td.e, strengths, grid_ev, fwhm_ev=0.3)`
returns a unit-area Gaussian oscillator-strength density per eV. This is a
Gamma-point per-cell convention; bulk k-integrated optical properties,
macroscopic absorption coefficients, dielectric q->0 limits, finite-q response
and spin-flip response are not implemented. Compare degenerate-state strengths
as group sums, since individual eigenvectors inside a degenerate subspace are
not uniquely defined.

For pure LDA/GGA, `energies, coefficients = mf.get_bands(kpts_path)` evaluates
arbitrary path k points using the converged SCF density. It preserves the
reference and processes queries in chunks (`chunk_size=4` by default).
Query k points have shape `(nquery,3)` in inverse Bohr; restricted band energies
have shape `(nquery,nmo)`, unrestricted energies `(2,nquery,nmo)`. HF/hybrid
band queries are explicitly unsupported until their exchange treatment at
arbitrary query k points is implemented.

`integrals.periodic.fft.build_inputs` accepts explicit dynamic basis parameters
and lattice arrays on a fixed FFT topology. Functional SCF entry points use
`SCFDifferentiationConfig` for implicit/unrolled response. First coordinate
derivatives have been tested for Gamma and two-k-point HF; periodic stress,
basis optimization and excited-state derivatives are not yet validated.
Gamma caches AO-pair potentials, with memory proportional to `ngrid * nao**2`;
k-mesh exchange processes k pairs separately. This initial backend has not been
benchmarked for large cells or GPU performance.

Run `PYTHONPATH=src JAX_PLATFORMS=cpu python examples/periodic_h2.py
--xc pbe --kmesh 2 1 1 --response full --mesh 41` for a reproducible example.

## UHF and UKS internal stability

SCF convergence does not certify a local minimum. For real UHF orbitals, use
`scf.uhf_stability(result, eri=eri)` to check an existing `UHFResult`, or
`scf.stabilize_uhf_from_integrals(**inputs)` to run SCF and follow negative
curvature with up to five directed restarts. `inputs` are the same arguments as
`run_uhf_from_integrals`; `max_restarts=0` requests analysis without restarting.

```python
outcome = scf.stabilize_uhf_from_integrals(**inputs)
print(outcome.result.total_energy, outcome.stable, outcome.restarts)
print(outcome.minimum_curvature, outcome.stability.residual_norms)
```

The check reuses JAX orbital-energy HVPs and the Davidson solver, with no dense
Hessian. Default internal curvature tolerance is `1e-5` Hartree per squared
orbital angle; eigenpair residual tolerance is independently `1e-7`. `stable`
is `None` when SCF or the eigenproblem did not converge, `False` when negative
curvature remains, and `True` when the check passes. Accepted restarts must
converge and lower the energy; the last accepted result is retained on failure.
This is an explicit host-side branch selection, not a differentiable restart
loop. Differentiate the selected branch with the existing SCF AD interfaces.
Internal stability does not prove a global minimum or external stability
against complex orbitals or UHF-to-GHF variations.

For UKS, use `scf.uks_stability(result, eri=eri, ao=ao,
ao_deriv1=ao_deriv1, grid_weights=weights, config=config)` or
`scf.stabilize_uks_from_integrals(**inputs)`. The latter accepts the inputs of
`run_uks_from_integrals` and uses one common stability/restart implementation
with UHF. Analysis must receive the same grid and `UKSConfig` as the SCF run;
XC density-floor settings are retained. HVPs include the XC spin response and
the specified exact-exchange contribution. Current support is real orbitals,
full ERIs and ordinary LDA/GGA/global-hybrid XC with unclipped potentials.
Density fitting, bound/Neural XC and external stability are not supported by
this entry point. A nonconverged UKS result returns an unknown stability status;
this operation does not replace a convergence solver.

## Shared SCF implementation boundaries

- `scf/convergence.py` owns finite-value checks and energy/density/orbital-gradient
  thresholds for RKS/UKS/ROKS/GKS. `conv_tol=0` explicitly disables early stopping.
  RKS now honors `conv_tol_density` and exposes independent `conv_tol_grad`.
- `scf/diis.py` owns the history ring, normalized Gram solve and extrapolation.
  Method-specific residuals, history shapes, cadence, occupations and diagonalizers
  remain separate; the existing RKS patch points are thin compatibility wrappers.
- `scf/energy.py` owns shared restricted/unrestricted energy and Fock algebra.
  `XCContribution` makes XC energy, potential, exchange fraction and extra Fock
  terms explicit. Bound adapters whose energy already contains exact exchange
  mark `energy_includes_exact_exchange` to prevent double counting.
- RKS has one internal array-valued `RKSResult` PyTree. `TraceableRKSResult` is a
  compatibility alias; only the eager public entry converts scalar fields to Python.
- RKS facades retain their completed result and integral/grid inputs. Response
  preparation supplements missing dipoles/features without rerunning SCF or S/H/ERI.
  Ground-state input/configuration changes require a new `kernel()` call; response
  feature changes reuse the same ground state. Cached inputs live until the next
  kernel or the facade is released.
- Former private `scf/features.py` helpers are test-only and now live in
  `tests/reference_scf_features.py`.

The XC adapter also preserves all three PW parameter sets when an upstream
`jax_xc` factory uses the legacy `get_p` conversion. It reconstructs parameter
vectors from the factory's actual arguments, retaining upstream formulas and
user overrides without modifying the installed module globally.

The recorded silicon/diamond band and silicon Gamma spectrum comparisons are
available in [periodic validation artifacts](reproducibility/periodic/2026-09-13/README.md),
including numerical arrays, figures, tolerances, and execution metadata.
