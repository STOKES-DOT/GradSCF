# XC locality and execution — fixed TDD scope

Baseline: `20a47391`, existing isolated worktree `/Users/jiaoyuan/.codex/worktrees/5fbb/GradSCF`.
The user approved these three axes on 2026-11-09. This is a new bounded round;
it does not relabel the older HFKS reference-recovery gates as completed.

## Contracts

- Preserve quadrature points/weights, float64, full orbital space and existing
  SCF thresholds. Conventional and true-RI consumers share the XC implementation.
- First establish exact reordering and density-representation equivalence.
  Hard AO screening is an explicit, default-off approximation; a local AO
  bound is not a global energy/error certificate. Report cutoff and overflow.
- Keep energy, potential and response on the same chosen definition. Geometry,
  centers, exponents, coefficients and model parameters stay live. Dynamic
  support must be recalculated; no frozen numerical cache or new stop_gradient.
- Preserve general density/response inputs. Occupied orbitals are acceleration
  hints only when they represent the current density. Their shortcut must
  retain arbitrary density JVP/VJP, not only the constrained orbital tangent.
- Standard additive local XC may use local blocks. Opaque/global custom or
  neural models retain their semantics and original grid ordering.
- CPU/GPU may choose different plans. Compare preparation + complete XC + SCF,
  not isolated kernels alone. No broad tuning framework or replacement solver.
- No production implementation before a focused remote RED. Keep changes small;
  retain an optimization only with measured benefit or explicit correctness/
  memory value. No merge or push in this round.

## TODOs

### XC-01: spatial organization and local AO work

- [x] 01a Exact permutation/inverse and derivative-preserving coordinates/weights.
- [x] 01b Conservative dynamic AO derivative bounds, actual compact AO inputs,
      padded capacity and correct dense overflow; zero/negative weights included.
- [x] 01c Connect local density and potential contraction to known additive
      consumers; retain general/global paths and explicit approximation controls.
- [x] 01d Validate values, JVP/VJP/HVP, two-step FD, updated geometry/basis state,
      cutoff convergence and actual reduction in evaluated AO dimensions.

### XC-02: conditional density representations

- [x] 02a RED fixtures for matching/mismatching MO hints, occupied tail, dtype,
      nonsymmetric/general densities, LDA/GGA/MGGA and dynamic parameter updates.
- [x] 02b A shared conditional helper with correct full density response and
      higher-order AD; no duplicate per-facade field formulas.
- [x] 02c Integrate hints once per physical state, including initial/mixed density
      safeguards, restricted/unrestricted/generalized existing support boundaries.
- [x] 02d Measure on realistic shapes and retain appropriate device/rank choices;
      compare standalone density kernels separately from complete XC and SCF.

### XC-03: cache, tile and device plan

- [x] 03a Reuse existing memory ownership and reserve rules; account for local
      work, possible dense overflow, hints and retained data without new budgets.
- [x] 03b Choose a small number of deterministic tile/capacity/layout policies
      using measured cases; avoid uncontrolled dynamic shapes/recompilation.
- [x] 03c Include construction/amortization and CPU/GPU transfer in comparisons;
      preserve compiled fusion and keep diagnostics separate from clean timing.
- [x] 03d Retain only justified choices, with explicit limits rather than a
      single unvalidated all-device threshold.

### XC-04: integration and delivery

- [x] 04a Small RKS/UKS/GKS + supported functional-family regressions; conventional
      and true-RI references, independent energies and original-Fock residuals.
- [x] 04b Existing custom/neural training modes, NNAO and supported geometry/
      mixed response regressions; no newly unsupported existing default path.
- [x] 04c CPU/GPU beta-carotene/def2-SVP/grid3 PBE0 anchor: clean total XC,
      preparation, SCF, actual dimensions/overflow, source hashes and memory.
- [x] 04d Independent review, remove failed candidates, document measured results
      and capability boundaries; local focused commits only.

## Acceptance and resources

Exact rearrangement values: atol/rtol1e-10; array AD1e-9. Two-step directional
FD normally <=1e-7, retaining tighter existing tests. Screened molecular total
energy <=1e-7 Ha, original-Fock residual <=1e-7 Ha, supported nuclear gradient
<=1e-5 Ha/Bohr. Screening defaults cannot be promoted merely from a small case.
Pure point-functional cost is not a target: prior measurements already isolate
0.291s CPU /0.0147s GPU versus complete XC10.120/2.292s on this anchor.

All scientific runs/builds on `jy_gpu_2` under Slurm: at most one NV5090,
64 logical CPU,256 GiB, heavy work serial;6000s individual guard,2h rounds,
200GiB RSS guard. Read live manual and allocations before each submission.
Local work is limited to edits, standard-library metadata and syntax review.

## Evidence log

- 2026-11-09: No active prior project allocation. Production source clean at
  baseline; unrelated benchmark/report edits remain preserved. First work is
  density-route and locality contract tests. No performance gain claimed yet.

- 2026-11-09 final review: a real zero-weight quadrature point must not be
  conflated with tail padding. Its weight derivative can be nonzero. The
  candidate is held before paired timing; the ongoing frozen-baseline process
  continues. New RED/GREEN will introduce explicit point validity. An older
  fixture that encoded padding only by weight zero will be updated to mark
  that padding explicitly; no numerical tolerance or derivative gate changes.

- 2026-11-09 14:12 +08: final zero-weight TDD is RED12 -> CPU71/GPU15
  GREEN. Final chosen fixed-density XC passes registered energy/potential
  gates on both devices. CPU cache9.93836s -> local6.54966s; GPU raw2.28141s
  -> local1.03404s. Compilation, sorting and actual-grid support are separate
  records. Complete SCF is still pending; these are operator numbers.
- Source freeze verified against `paired_ready_zero_weight_fix.json`; batch
  118316 resumed after the final kernel checks. Baseline CPU4/4 accepted,
  candidate CPU and paired GPU calculations now proceed serially.
- Checked01/02/03/04 items refer to the explicit small regression and
  fixed-density scopes in the report. 02d awaits whole-MGGA confirmation;
  04c/04d await paired full SCF, final review and local delivery.

- Final acceptance 2026-11-09:16/16 full PBE0 SCF calls passed. CPU warm
 454.5869 ->301.3738s; GPU222.8001 ->182.2576s. Whole-r2SCAN XC at actual
 saved PBE0 states passed both exact guards and potential/energy gates, with
 CPU30.4291 ->19.6127s and GPU8.2773 ->5.4148s. Source/spec and quality
 reviews passed; production commitdb9700ff. Resource jobs have ended.
- Automatic MO size/rank eligibility is explicitly a heuristic. Threshold
 neighborhoods/high occupancy remain unmeasured; no universal optimum claimed.
- This checklist is closed for its stated scope; older HFKS reference-recovery
 gates are unchanged. Final report and committed reproducibility data provide
 scope, counterexamples, failure evidence and numerical/AD boundaries.
