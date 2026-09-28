# External neural XC functionals

`ExternalFunctional` connects an external feature builder and external network
energy to the existing GradSCF SCF, AD, response and training code. It does not
choose the network architecture, prescribe a feature vector, or add an XC
baseline automatically. It introduces no separate SCF solver or trainer.

```python
from gradscf.model.neural_xc import ExternalFunctional

functional = ExternalFunctional(
    input_fn=make_inputs,         # DensityInputs -> array PyTree
    energy_fn=energy,             # (params, inputs) -> scalar XC energy, Ha
    init_fn=initialize,           # optional (key, inputs) -> parameter mapping
)
```

`make_inputs` and `energy` must be JAX differentiable to the order required by
the requested loss. Flax, Haiku or pure-JAX networks can be called inside
`energy`; no inheritance from a GradSCF neural network is required. Trainable
weights are explicit parameters, not mutable object state or captured traced
closure variables. The current Flax TrainState-based trainer expects a mapping
at the parameter root; nested parameter values can be PyTrees. Supply existing
weights using `NeuralXCTrainer.kernel(params=...)`, or provide `init_fn`.

## SCF-current physical inputs

For every trial density D, GradSCF constructs a `DensityInputs` view. The
external input function is evaluated **inside** the XC energy derivative:

```
trial D -> DensityInputs -> make_inputs -> network energy -> dE_xc/dD
```

Thus derivatives include the feature construction and features are updated
on each SCF evaluation. There is no cross-iteration density-feature cache.
The view reuses existing contractions in `gradscf.tools.features`; quantities
are computed on access, so density-only networks do not require AO gradients.
Missing AO derivatives/Laplacians raise an error rather than supplying zeros.

| Property | Meaning and layout |
|---|---|
| `density_matrix` | Current symmetric alpha/beta matrices, `(2, nao, nao)` |
| `total_density_matrix` | Spin sum, `(nao, nao)` |
| `rho`, `rho_spin` | Total `(ngrid,)` / spin `(2, ngrid)` density |
| `grad_rho`, `grad_rho_spin` | Total `(ngrid, 3)` / spin `(2, ngrid, 3)` gradient |
| `tau`, `tau_spin` | Total / spin kinetic-energy density; tau = 1/2 sum occupied orbital-gradient squares |
| `laplacian_rho`, `laplacian_rho_spin` | Total / spin density Laplacian |
| `weights`, `coordinates` | Grid weights and coordinates |
| `ao`, `ao_deriv1` | AO values; AO values plus three first derivatives |
| `atom_coords`, `atom_charges`, `overlap_matrix` | Geometry, charges and AO metric |

All quantities use atomic units. Density-independent arrays are reused at
fixed geometry; if geometry/grid/AO data are differentiated, their input
arrays must remain on the JAX graph. The feature builder may produce nested
dictionaries/tuples and may couple grid points. Use differentiable smooth
operations where potentials or higher responses are required.

Canonical orbital coefficients/energies, frozen reference HFX and PT2 caches
are not exposed as density ingredients. Treating them as independent constants
while varying D would omit their response. Arbitrary architecture does not
mean an arbitrary black-box runtime or arbitrary orbital-dependent physics.
Complex/current-dependent functionals and a separate exact-exchange fraction
require an explicit adapter with the corresponding physics; they are not
inferred by this real-density interface.

## One energy, potential and response

`energy_fn` returns the **total scalar XC energy**, not per-electron energy,
energy per volume, a potential, or network coefficients. For per-electron
network output epsilon, explicitly return `sum(weights*rho*epsilon)`. For a
learned correction, explicitly add the baseline in that same energy function.

- `energy_for_density(params, molecule, density)` evaluates trial D.
- `energy_from_molecule(params, molecule)` uses the state's current D.
- `potential(params, molecule, density)` calls the shared
  `scf.xc_energy_and_potential_from_density`.
- `kernel_action(params, molecule, density, tangent)` calls the shared
  `scf.xc_kernel_action`, a JVP of the potential. The tangent has D's shape.

The kernel is an AO density-matrix Hessian action, including external feature
and network derivatives. It is **not** a scalar pointwise f_xc approximation;
nonlocal grid coupling is retained. No complete four-index Hessian is built.
Both spin-stacked and spin-summed densities are supported. TDA/TDDFT use the
same adiabatic real-density Hessian through their existing AO transition and
projection routines. Parameter derivatives remain available through the kernel.

## Existing training modes

Use the existing `MolecularTrainingDatum`, `MolecularTrainingConfig` and
`NeuralXCTrainer`; no external-network-specific training loop is introduced.

| Mode | Configuration |
|---|---|
| Fixed density | `mode='fixed_density'` |
| Unrolled SCF | `mode='self_consistent', scf_gradient_mode='unrolled'` |
| Implicit SCF | `mode='self_consistent', scf_gradient_mode='implicit'` |

Fixed-density energy evaluation differentiates at the supplied state. A
self-consistent objective runs the existing SCF with the current weights.
Density-specific losses retain their existing behavior of requesting a
self-consistent density. Unrolled AD differentiates the executed iterations;
implicit AD uses the existing fixed-point adjoint and shared linear solver.

For external functionals, SCF training requires forward convergence by
default in both backward modes. Set `scf_require_converged=False` explicitly
to train finite, unconverged unrolled iterates. `scf_require_converged=True/False` explicitly controls that gate;
`None` uses the functional's policy (legacy functionals retain their existing
default). The implicit derivative requires a locally differentiable isolated
fixed point; convergence alone does not prove stability or a global minimum.

The molecular train step rejects nonfinite loss/gradient updates and preserves
both parameters and optimizer state. `update_accepted` and
`nonfinite_grad_fraction` are retained in the `NeuralXCTrainer` history. The
low-level loss helper still returns sanitized gradient arrays for compatibility;
its nonfinite metric must be checked by callers writing their own optimizer
loop. The XC energy-derivative helper no longer converts an invalid potential
into zeros. Other legacy XC paths are unchanged.

## Executed demonstration and validation

[external_functional.py](../../../../examples/neural_xc/external_functional.py)
uses external Flax features/architecture and GradSCF H2/6-31G* native integrals.
It adds a neural correction to Dirac exchange, with correlation omitted.
Targets come from a second parameter set and are a training demonstration,
not a chemical-accuracy benchmark. No CLI or PySCF calculation is involved.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/neural_xc/external_functional.py
PYTHONPATH=src JAX_PLATFORMS=cpu \
  python -m pytest -q tests/test_external_neural_xc.py
```

Measured on 2026-09-28, Apple M4 Pro CPU, float64, JAX 0.8.1, seed 0.
Three Adam steps, learning rate 0.002, SCF maximum 80 cycles, density tolerance
1e-9, energy tolerance 1e-11 Ha, implicit tolerance 1e-9.

| Mode | Initial energy MSE / Ha² | Final energy MSE / Ha² | Seconds |
|---|---:|---:|---:|
| Fixed density | 4.1417802781e-6 | 9.2387785597e-7 | 1.69 |
| Unrolled SCF | 5.6129071029e-6 | 1.6771517640e-6 | 4.80 |
| Implicit SCF | 5.6129071029e-6 | 1.6771517640e-6 | 8.23 |

Timings include different compilation/cache states and are not a performance
comparison. CPU tests check current-density features, nonlocal feature
coupling, potential/HVP finite differences, mixed parameter/kernel derivatives,
R/U SCF density responses, three-mode training, TDA/TDDFT A/B actions,
initialization, and failure rejection. GPU and complex-orbital execution have
not been validated.

Validation scope: the initial expanded run recorded 102 passed, 2 skipped,
12 failures caused by missing jax-xc, and one deliberately deselected legacy
orbital-energy fixture failure. All 12 dependency failures and the legacy
fixture failure were independently reproduced on unchanged base fced3c4.
The legacy fixture lacks bind_to_molecule_for_scf. They were not repaired or
reclassified as passing by this change. The final targeted results are listed
below; no full-repository or GPU validation is claimed.

Final targeted run after review corrections: **54 passed, 1 deselected**,
58.59 seconds, CPU float64. The deselected case is the independently confirmed
legacy fixture failure above. The external-functional suite alone has 15 tests.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_external_neural_xc.py tests/test_scf_implicit_and_xc_energy.py \
  tests/test_molecular_training_api.py tests/test_scf_autodiff.py \
  tests/test_training.py \
  --deselect=tests/test_training.py::test_orbital_energy_loss_uses_explicit_target_and_weights
```
