# Analytic XC design with a fixed Python evaluator

The Python controller owns the experiment. An optional Codex thread proposes a
formula; GradSCF evaluates it; the controller records the result and feeds back
aggregate diagnostics. No provider RPC implementation, handwritten PBE,
independent eigensolver or dataset-specific workflow is copied into this package.

| Module | Responsibility |
| --- | --- |
| `candidate.py` | Immutable formula/parameter contract, safe arithmetic parser, JSON schema and digest |
| `codex.py` | Optional official `openai_codex` thread, structured replies, resume and transport timeout |
| `loop.py` | Shared AD parameter fitting, fixed-batch evaluation, acceptance and feedback |
| `history.py` | Source/configuration fingerprint, append-only records and atomic proposal checkpoints |
| `dft/iks/wu_yang.py` | Physical fixed-target score and shared occupied-subspace response |

## Physical evaluation

```python
problem = iks.WuYang(target, kinetic, base_fock, baseline=baseline_vxc, grid=grid)
score = problem.score(candidate_vxc)
# Alternatively: complete scalar XC energy(params, D), including any baseline.
score = problem.assess(xc_energy_fn, params)
```

`base_fock` is **T + Vext + J[Dtarget]**. Both `baseline` and the argument of
`score` are **complete XC potentials**, added exactly once. The target AO
convention, overlap, grid and all matrices must match. Cartesian embeddings of
spherical datasets must not enlarge the orbital variational space.

RKS uses the spin-summed density and occupation two; UKS keeps fixed alpha/beta
counts and independent constant gauges. Complex/nonlocal GKS is explicitly
outside this local Wu–Yang interface; the older fixed-reference GKS assessment
remains separate. All quantities are in atomic units. For candidate F = base+Vxc,

    Es = Tr(F Dcandidate)
    W  = Es - Tr((F-T) Dtarget)
    G  = Tr(F Dtarget) - Es
    reward = (W - Wbaseline) / (Ne * energy_scale)

Delta W is computed through spectral-energy differences and the candidate XC
change to reduce cancellation. `energy_scale` defaults to 1e-3 Ha. It is a
normalization choice, not a physical accuracy threshold. Internal degeneracy is
allowed; an occupied/virtual gap must exceed `gap_tolerance` (default 1e-7 Ha).
The shared solver handles subspace response; no single-orbital AD is introduced.

Preparation owns its baseline and overlap transform. There are no process-global
or object-id caches. `score` and `assess` retain JAX parameter derivatives;
Python/JSON conversion happens only at the reporting boundary. An invalid
candidate has `valid=False` and NaN reward. Optional relative grid density error
is reported independently; unrequested error is NaN, not a measured zero.

For an external local field, `problem.local(field_fn, params)` reuses the existing
blocked `iks.Potential`. The callback receives parameters and coordinates and
returns `(points,)` RKS or `(2,points)` UKS potential values. Its external input
provider evaluates descriptors at the fixed target. An AOGrid in the identical
target AO convention is required. Parameter-affine AO directions can instead be
prepared once in `bind`; arbitrary nonlinear fields are not silently cached.

This score needs neither inverse orbitals nor a teacher potential. It does not
run candidate SCF, certify a unique continuum potential or assess the final SCF
density. The older `iks.reward` and `iks.invert` retain their existing contracts.

## Candidates and existing custom_xc

```python
candidate = autoxc.Candidate(
    'a*n**2', kind='energy',
    parameters=(autoxc.Parameter('a', 0., -.5, .5),),
    hypothesis='Test a small density-dependent energy correction.')
functional = candidate.functional(input_fn)
```

An energy candidate evaluates an energy density, integrated with explicitly
supplied `weights` through the existing `dft.custom_xc`. `input_fn` receives
`dft.DensityInputs`, so features remain external and depend on the trial density.
Neural or traditional scalar energy callbacks can go directly to `assess`;
they need not be serialized as analytic candidates. Existing Trainer modes are
unchanged. In particular, design rounds do not add a fourth training mode.

Potential-only candidates define a local potential in Ha. They are evaluated
through `score` or `local`, and are not assigned an invented XC energy. No hidden
baseline, clipping, floor, spin convention or regularization is inferred. Feature
units, domain and any smooth vacuum treatment belong to the declared input/model.

The parser permits scalar arithmetic, bounded literal powers and a small set of
JAX math functions. It rejects Python execution, attributes, indexing and imports.
Formulas are limited to 120 nodes, depth 12 and four bounded fitted parameters.
These are pilot limits, not claims about the best functional family.

## Deterministic evaluator and loop

`bind(candidate)` returns a fixed mapping from sample IDs to `score_fn(params)`.
Each score function must return `iks.WuYangScore`. `Evaluator` fits coefficients
with JAX value/grad and the shared bounded `minimize_eager` solver. Any invalid
sample invalidates the whole candidate; no failed molecule is omitted to improve
the mean reward. Failed fits never become negative-infinity optimization targets.
An optimizer iteration limit is retained in the fit report, not called convergence.
An optional `score_fn.diagnose(params)` runs expensive grid diagnostics once
after fitting. Its W/DeltaW/reward must agree with the matrix score, and its
validity also gates the final candidate. The fit never repeats that grid pass.

```python
evaluator = autoxc.Evaluator(bind, sample_ids=train_ids + validation_ids,
    fit_ids=train_ids, reward_ids=validation_ids, fingerprint=experiment_digest)
result = autoxc.run(designer, evaluator, directory='runs/xc-design', rounds=5,
                    context=design_brief)
```

`fit_ids` and `reward_ids` are explicit and frozen. They default to the complete
batch, supporting training-reward selection as well. Sealed tests must remain
outside this evaluator and all proposal feedback. After freezing the winner,
the caller may evaluate it once on a separate held-out batch without refitting.
No split or fixed list of molecules is embedded in the library.

`fingerprint` must identify the dataset/target, AO and grid conventions, callback
code, baseline, score scales and physical gates. The controller additionally
hashes the complete GradSCF Python source tree (including indirect shared
numerical dependencies) and records numerical/SDK versions and JAX's default
backend/float64 setting. A changed experiment
requires a new run directory. The caller remains responsible for updating an
external callback/data fingerprint when its semantics change, including native
backend builds not represented by Python source. Native binaries and raw data
are not read or copied by the source fingerprint.

The JSONL ledger retains every attempt, raw reply, exact formula/bounds,
fitted parameters, per-sample W/DeltaW/G/gap/error/validity, fit status, errors,
elapsed time and thread ID. An SDK error is a proposal failure, not a low score.
Feedback contains only formula/aggregate results; raw matrices and sample IDs
stay in local records. Checkpoints are atomic; completed replies survive an
interruption during evaluation. Restart with the same directory and a larger
**total** round limit. Completed rounds are not repeated. A reply not yet saved
when the process dies may need another model turn. Use one controller per run
directory; concurrent writers and automatic repair of a torn ledger are unsupported.

## Codex and verification boundaries

The optional `autoxc` extra installs `openai-codex>=0.147.0` and SciPy. SDK imports
are lazy. Creating/importing `CodexDesigner` does not authenticate or launch a
runtime; entering its context does. The SDK owns its pinned runtime and normal
authentication. No credential discovery, custom transport or endpoint probing
is implemented here.

```python
with autoxc.CodexDesigner(cwd=proposal_directory, model=model, effort='high') as designer:
    result = autoxc.run(designer, evaluator, directory=run_directory, rounds=5)
```

The proposal thread is read-only. That prevents evaluator edits, but does not
itself prevent reads of accessible held-out data: deployment must isolate such
data from the SDK process. A watchdog closes the official SDK runtime on a turn
timeout, preserving its thread ID; it uses no Unix signal handlers. A restart
restores the saved session through the SDK. An unfinished model turn and the
controller's last committed numerical result remain distinct states.

`examples/iks/agent_loop.py` runs three replayed proposals on a manufactured
two-AO quadrature. Fits and scores are real calculations. This is an executable
interface/restart example, not a molecular accuracy result or a live agent search.
The current verification uses CPU float64, fake SDK clients and array/quadrature
systems. Live SDK authentication, molecular XC100 runs, GPU execution and process
isolation enforcement are not claimed by those tests.

Subsequent acceptance now includes [one actual XC100/Codex loop](../../../../examples/iks/XC100_CODEX_SINGLE.md):
BH3 at the original cc-pVTZ/grid-5 settings, two live Luna/high proposals and
independent PBE checks. The original manufactured tests and that molecular
receipt remain separate. No generalization, GPU or candidate SCF claim follows.
