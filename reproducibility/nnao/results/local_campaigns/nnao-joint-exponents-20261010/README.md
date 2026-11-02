# NNAO contraction and exponent pilot

This experiment alternates shared MACE contraction training and derivative-free
Gaussian exponent updates. It starts from the frozen 24-structure epoch-975
checkpoint and optimizes only four original training structures: CH4, NH3, H2O
and HF. The independent ten-structure benchmark is not optimization data.

## Parameters and objective

The objective is the equal-weight mean of re-converged all-electron DF-RHF total
energies in Hartree. Core6, atom positions, contracted AO count and the
`def2-universal-jkfit` auxiliary basis are fixed. The orbital basis is
`szp663_direct`, spherical, float64.

Fourteen shared log-scales control the nonfixed shells:

```
H: s, p
C, N, O, F: s, p, d
alpha(Z,l,p) = alpha_original(Z,l,p) * exp(beta(Z,l))  # alpha in bohr^-2
```

The same beta applies to every atom of the relevant element. Each candidate is
constructed from the original exponent pool, so scales do not accumulate.
Within-shell exponent ratios remain fixed. Bounds `[-0.25, 0.25]` correspond to
multipliers approximately `[0.779, 1.284]`.

## Alternating updates

1. At fixed exponents, update the MACE weights with Adam using the existing
   implicit SCF coefficient gradients. Initialize new Adam moments for this
   four-structure experiment.
2. Freeze those weights and run bounded Powell searches one element at a time.
   Recompute only molecules containing that element. Each function evaluation
   re-converges SCF; no exponent derivative is requested.
3. For accepted exponent changes, rebuild the affected primitive S/H/DF caches,
   compare their projected energy with native contracted DF energy within
   `1e-8 Ha`, and validate the complete implicit backward before committing the
   joint state. Continue coefficient training at fixed new exponents.

SCF convergence, the raw Aufbau fixed-point residual, orbital residual,
overlap conditioning and finite gradients remain acceptance requirements.
Invalid trials cannot overwrite accepted parameters or Adam state. A Powell
evaluation-budget stop is recorded separately from optimizer convergence.

## Integral execution and isolation

Native CPU integrals provide S/H and packed three-center factors. A fixed
auxiliary Coulomb metric factorization is reused; candidate evaluations request
new 3c integrals only. MACE, projected DF, SCF and implicit backward run on the
allocated GPU. No complete four-index ERI or production PySCF call is used.

The formal experiment snapshot is based on remote scientific source
`87f14784...`, with only the declared basis/DF/experiment changes and the new
joint trainer. Each job copies that snapshot, audits original and experiment
hashes before and after execution, and records its own outputs. The original
200-structure training checkout and frozen initialization checkpoint remain
unchanged. This pilot does not implement native exponent automatic
differentiation or establish held-out generalization.

## Verification and results

Local CPU checks: 98 passed, one skipped because the local MACE construction
dependency `cuequivariance` is unavailable. These include scaled-basis implicit
coefficient finite differences, metric-cache parity and existing coefficient
and projection-AD regressions, plus eleven joint-controller tests. For scaled
H2, the coefficient directional gradient differs from fourth-order finite
differences by `1.0446e-11 Ha` at step `1e-4` and implicit tolerance `1e-8`;
see `local-h2-gradient.json`.

The remote pilot is Slurm job `95164` on `n002`, one V100 and sixteen CPU cores,
128 GiB requested memory, thirty-minute limit. It runs two Adam steps before
and after one sweep, learning rate `1e-5`, at most thirty Powell evaluations per
element, and implicit tolerance `1e-8`. Completed results are archived under
`job-95164/`; numerical completion is checked independently from job launch.

```bash
python tools/train_nnao_joint.py \
  --structures train-pilot.jsonl --checkpoint /path/to/original/best.npz \
  --output-dir /path/to/fresh/results --cache-dir /path/to/fresh/cache \
  --coefficient-steps 2 --cycles 1 --maxfev 30 \
  --learning-rate 1e-5 --log-bound .25 --implicit-tolerance 1e-8
```

The current pilot entry point requires the exact original best975 checkpoint
and original training IDs; it is not yet a general checkpoint-resume trainer.

## Completed four-structure result

Job `95164` finished on **2026-10-10 at 12:26:31 +08:00**, with
`phase=complete` and `exit_code=0`. The controller elapsed time was
**257.419837 s**, including its initial cache preparation and first
compilations; model/auxiliary setup and source auditing occur outside that
controller timer. Hardware was Tesla V100-SXM2-32GB on `n002`, with sixteen
allocated CPU cores. JAX/JAXlib were `0.10.0`, SciPy `1.17.1`, Flax `0.12.7`,
Optax `0.2.8`, e3nn-jax `0.21.0`, and mace-jax `0.2.0`.

| Accepted stage | Mean DF-RHF energy / Ha | Stage decrease / mHa per structure |
|---|---:|---:|
| Original best975 weights, original exponents | -68.051431079917 | — |
| Two coefficient steps | -68.051459926428 | 0.028846511 |
| H exponent update | -68.052102555301 | 0.642628873 |
| C exponent update | -68.052176079315 | 0.073524015 |
| N exponent update | -68.052791304602 | 0.615225287 |
| O exponent update | -68.053196465755 | 0.405161153 |
| F exponent update | -68.053716753922 | 0.520288167 |
| Two more coefficient steps | -68.053749733485 | 0.032979563 |

Total mean decrease: **2.318653568 mHa per structure**. Exponent updates
contributed **2.256827494 mHa** and coefficient updates **0.061826074 mHa**.
Initial per-structure energies were not stored in this run's history; do not
derive per-structure improvements by combining older-source tables.

| Element | s multiplier | p multiplier | d multiplier |
|---|---:|---:|---:|
| H | 1.043154402 | 1.244487626 | — |
| C | 1.003500041 | 0.972637267 | 0.976868534 |
| N | 0.990990239 | 1.014498978 | 0.778800783 |
| O | 1.011609927 | 1.001416565 | 0.778800783 |
| F | 1.002585696 | 1.018027767 | 0.778800783 |

All five searches used thirty function evaluations and ended
**budget-limited**, with a valid improvement accepted. There were 150 valid
objective calls, zero invalid calls and five cache hits; 272 molecular native
contracted evaluations include search trials and parity validation. N/O/F d
scales reached the `-0.25` lower bound. This is one bounded optimization pilot,
not an unconstrained exponent optimum or held-out accuracy result.

The final four SCFs converged without rescue, with at most twelve iterations.
Maximum final native-contracted/projected-primitive energy difference was
**3.55e-14 Ha**; maximum orbital residual **2.322e-10**, maximum raw fixed-point
residual **1.183e-10**, and minimum overlap eigenvalue **0.105591587**. The
7,171 weights, fourteen scales, 7,171-component gradient and Adam moments were
all finite; Adam count was four. These maxima apply to the final checkpoint.

Timing totals (all systems and applicable cold compilations included): native
contracted integrals **60.682 s** over 272 calls; primitive-cache integrals and
writing **7.576 s** over twelve constructions; primitive SCF/implicit
**158.717 s** over forty calls; NN VJP **11.054 s** over forty calls. These are
phase aggregates, not a steady-state speedup against the old 24-structure run.

Evidence: `job-95164/training/summary.json`, `training/candidate_history.json`,
`training/last.npz`, and `provenance-before.json` / `provenance-after.json`.
Both provenance records match, verify 764 baseline files, and show the original
training source/checkpoint unchanged. New scientific source hash is
`c15a3cc5cf5f6c4c9f2b538534693d2dad2e4f0544357b664e1fad43b97edb41`;
trainer hash `c029d23038718a0fc8f0eb5df2a33438461288bbd5d83ea4f6e56650a4bb3ace`;
accepted checkpoint hash
`8da6e63d1f6a5e365c93f32dfc533fa49d3f60e5631dd45082d86acc1d3d30c5`.
