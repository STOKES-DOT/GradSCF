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
- differentiable explicit (`expl`) and implicit (`impl`) SCF modes;
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

The `upstreams` extra installs PySCF and `jax-xc`. For manuscript scripts and
checkpoint evaluation, install:

```bash
python -m pip install -e ".[dev,reproducibility]"
```

GPU runs require a CUDA-enabled JAX build and a GPU4PySCF installation matched
to the CUDA environment. Confirm the active backend before a long run:

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
under `gradscf.integrals.assembly`; `scf.inputs` and `data.integrals` are temporary
compatibility imports pointing to the same implementations.

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
ERI, and `jax.jit`. **Native JVP/VJP is not yet implemented and explicitly
raises**, without a hidden fallback. Use `backend="jax_reference"` for the
current Cartesian basis-parameter derivative path. First derivatives of small
systems are tested; higher derivatives are not advertised. Inspect
`integrals.backend_capabilities(...)` for backend-specific contracts.

This new native plan is opt-in. Existing SCF `integral_backend="cpu"` retains
its previous PySCF/libcint path during migration. See the
[native backend guide](src/gradscf/integrals/_native/README.md)
for pinned sources, build requirements, ABI, and supported operators.

Integral implementations and build resources are grouped under
`src/gradscf/integrals/`: public APIs and assembly at the package root,
backend adapters in `backends/`, and the private loader, C/C++ sources,
vendored dependencies, and CMake files in `_native/`. The old
`python -m gradscf._native.build` command remains a compatibility entry point.

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

Use `integral_backend="gpu"` and `execution_device="gpu"` in a configured
GPU4PySCF environment. `integral_backend="cpu"` uses the CPU integral path;
the resulting fixed molecular integrals are reused by SCF and response calls.

A complete PySCF-versus-GradSCF B3LYP comparison is provided in
`examples/compare_pyscf_vs_jax_tddft_no_neural.py`.

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
from pyscf import dft as pyscf_dft, gto as pyscf_gto
from gradscf.data.reference import restricted_reference_from_pyscf

pyscf_mol = pyscf_gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="def2-svp",
    unit="Angstrom",
    verbose=0,
)
pyscf_mf = pyscf_dft.RKS(pyscf_mol)
pyscf_mf.xc = "b3lyp"
pyscf_mf.grids.level = 2
pyscf_mf.kernel()

reference = restricted_reference_from_pyscf(
    pyscf_mf,
    compute_local_hfx_features=True,
    compute_local_hfx_aux=False,
    compute_local_pt2_features=False,
    jk_backend="full",
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
`examples/h2_fci_self_consistent_train.py`. Production H2, H2+, N2, QM9, and
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
| Conventional PySCF validation | `tools/compare_pyscf_vs_jax_same_xc.py` |
| Benzene Davidson convergence | `tools/trace_benzene_pbe_tddft_davidson.py` |
| H2+ ground-state dissociation | `tools/h2plus_fci_ground_train5_dense100.py` |
| H2 ground-state dissociation | `tools/h2_self_consistent_ground_train5_dense100_vs_fci.py` |
| N2 ground-state dissociation | `tools/n2_ccsdt_ground_train5.py` |
| H2/N2 S1 TDA dissociation | `tools/h2_s1_tda_train5_dense100_vs_fci.py` |
| QM9 ground and QM9GWBSE S1 training | `tools/closed_shell_s1_self_consistent_train.py` |
| Released checkpoint inference | `tools/evaluate_closed_shell_checkpoint.py` |
| QM9/QM9GWBSE baselines and figures | `tools/compute_qm9_ground_classic_baselines.py`, `tools/compare_qm9_pyscf_vs_jax_tda.py`, `tools/plot_qm9_reference_structures.py`, `tools/plot_qm9_val_bars_with_structures.py` |

Example paper commands:

```bash
python tools/h2_self_consistent_ground_train5_dense100_vs_fci.py \
  --basis def2-tzvp \
  --grids-level 2 \
  --ground-state-hf-mode nograd \
  --ground-state-pt2-mode off \
  --steps 2000

python tools/h2_s1_tda_train5_dense100_vs_fci.py \
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

Some reference comparisons require PySCF, GPU4PySCF, CUDA, or generated input
data. Large integral/basis sweeps are not part of the compact release gate;
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
Optax, `jax-xc`, PySCF, and GPU4PySCF. Third-party data or source snapshots keep
their original licenses and notices.
