# TDD fixture audit before implementation

## 2026-11-05: symmetric AO-pair parameterization

The first original-source run produced 27 failures and 2 passes in 13.56 s.
Its JUnit SHA256 is
`68bd41ce0ed123d720bbb1901d4c4c6a409717d16af994d58a86e323b0f1b771`.
Two packed cases passed values and symmetric-direction JVP, but their factor
VJP compared two different parameter spaces: packing reads the lower triangle,
while the dense oracle independently varied upper and lower entries.

Both actions and the oracle now explicitly parameterize symmetric spatial
AO-pair factors. The physical values, tolerances, complex-density dependency
and required VJP equality are unchanged. This is a fixture correction, not a
packed-kernel fix. Existing generic raw-factor regressions remain required.
The original logs and initial lock are retained; original-source tests are
rerun with the corrected fixture before generalized production changes.

Four restricted RI-J/direct-K tests were also registered before modifying
RKS or its public facade. They cover the public selection/config signature,
different Coulomb and exchange sources, full/LR weighting, and array JVP/VJP.
These instantiate the already registered T2.4 requirement; no physical case,
performance gate or numerical tolerance was relaxed.

## 2026-11-05: unrestricted dense-oracle output contract

The first UKS candidate run passed 11 tests and failed one VJP comparison.
Both the original and candidate production functions return symmetric J/K
matrices, but the fixture's dense oracle omitted that output projection. Its
random nonsymmetric loss weights therefore differentiated a different ambient
matrix function. For example, B=I and W[0,1]=1 give density gradients (0.5,0.5)
for Sym(D), whereas the unprojected oracle gives (1,0).

The oracle now explicitly symmetrizes its outputs. The raw factor/density
parameter spaces, random weights, test groups and every tolerance are retained.
AST comparison confirms that the original and candidate orbital-hint custom
JVP and dense DF contraction are unchanged. The original fixture and failed
candidate evidence are retained. The numerical gate remains open until rerun.

## 2026-11-05: HF reference compatibility gate

The existing T4.2 requirement also covers HF-derived inputs used by neural XC
and explicit HFX features. Six narrow tests confirmed that skipping the HF
grid without lazy reference completion breaks these consumers. They were RED
before the completion patch. This does not create a new optimization phase.

One compatibility fixture initially provided only density/convergence fields,
while the real reference builder also requires orbital/energy fields before
it reaches HFX. It now supplies a complete stationary two-electron RKSResult;
assertions and tolerances are unchanged. The initial fixture SHA256 was
`ce6e97dc90f987c74c67d685d5d2ad6a0d98049c4520ff81da6542d1bbb3de11`.
Original logs remain under `results/compat-green/` in the isolated remote run.

The existing GKS grid test formerly required rejection of streaming. It now
checks that explicit dense (-1), automatic (0), and streamed (2) requests are
transmitted unchanged, preserving the dense-mode test and adding the newly
authorized behavior. No numerical accuracy gate was removed.

The benchmark worker was audited before large execution: old repeat payloads
are released, actual grid size is reported, and RI-J/packed-K is distinguished
from RI-J/direct-K. These harness corrections do not change the physical
protocol or the sixty configuration hashes.

## 2026-11-05: manifest fingerprint audit

The final lock audit detected that the initial manifest file hash preceded
addition of its `config_file_sha256` map. Removing only that map reconstructs
the exact earlier bytes and SHA256
`9a00ff87fa6fd1d345c76b1ff897cc3a2a8d53baed78b61eae8628999a4a6ce5`.
The complete manifest SHA256 is
`3f11abf0e92247203444cbcabe89f81c1ccddbc2f0b3e4c01ec54901b74c5089`.
Every one of the sixty configuration files still matches its initial lock,
and the physical protocol and geometry fingerprints are unchanged. The lock
is corrected to include the stronger manifest, retaining its previous revision.

## 2026-11-05: independent reference certification and observer repair

The original PySCF single-kernel attempt remains immutable and unaccepted.
Its 24 main iterations passed the registered checks, but a subsequent extra
diagonalization returned a gradient of 2.217132e-7 Ha and revoked convergence.
PySCF 2.13 then returns rather than continuing the main loop. This is not
failure to converge within 200 main iterations.

An optional, explicitly labelled `--reference-continuation` policy uses the
same object and current density only after this specific final-check failure.
The initial guess, all three thresholds, conv_check and the physical inputs
remain unchanged. The total main-iteration budget remains 200; extra checks
are separately counted. A repeated kernel starts a new local DIIS history.
The original logical-call wall timer and 6000-s budget cover every restart.
This policy is a new reference variant, not a relabelled original baseline.

The original observer also called `DF.get_naoaux()` after the timed solver.
For pure RI-J, PySCF may not retain factors, and that query calls build(),
creating full three-center factors only for metadata. Thus the original
249-s whole-process time and RSS are contaminated, while the 174-s timed
facade remains a failed single-kernel measurement. The repaired observer
reads existing auxiliary-basis dimensions and existing factor shape only;
unknown effective rank remains null. No additional integrals are computed.

These controls were tested with stdlib fakes before the new reference run.
All sixty physical configuration hashes and numerical gates remain unchanged.

## 2026-11-05: T2.3 production AO reuse contract

Seven narrow tests confirmed three execution REDs before the new production
path: no funded call-local preparation, no repeated-Fock reuse, and no run
entry-point connection. Four pre-existing fallback/AD guards passed. Six
additional metadata-sized REDs covered known native/RI workspace reservations.
These tests instantiate T2.3/T3.2 without adding systems, changing quadrature,
or relaxing accuracy gates.

The derived AO reducer is local to builtin SCF. Raw providers, reference
objects and global caches are not mutated. Native/RI workspace is charged
before optional retention; UKS reserves two possible RI exchange products.
Unknown operators/devices retain streaming. This is a forward-array estimate,
not a hard bound on arbitrary AD tapes.

The AD guard was strengthened with two-step FD and a dense HVP. Instrumented
tests clear only the private materializer JIT cache between monkeypatched
observers; numerical gates and production compilation reuse are unchanged.

After the 880-AO prototype showed 106.7 s cache preparation, the same feature
was refactored to stable JIT and an exact output with disjoint block/tail writes.
Old prototypes and measurements remain archived. The refactor passed all 13
cache/budget/AD tests before optimized complete-workflow execution.

## 2026-11-05: prioritize conventional HF/DFT coverage

The user explicitly requested normal non-RI HF/DFT tests in addition to RI.
The existing sixty configurations, physical inputs and accuracy gates are
unchanged. A `--direct-first` execution overlay stably prioritizes the thirty
conventional rows while preserving each representation group's original order.
The default controller order is unchanged. Two metadata-only tests were RED
before adding the flag and then GREEN; they check all sixty hashes, ordering,
process caps and prior-attempt handling. Revision 7 of the lock is retained.

The priority complete workflows use the already registered RHF/direct inputs
for GradSCF and independent PySCF. Separate fixed-five RHF, SVWN, PBE0 and
CAM-B3LYP diagnostics use the same 96-atom basis and level-3 KS grid, followed
by J-only/K-only/combined products at one shared snapshot. Their comparison
tolerance is declared in `conventional_execution_overlay.json` before those
kernel measurements. These snapshots are deliberately unconverged and do not
replace the original complete-workflow convergence gates or warm repetitions.

The running conventional-PBE scientific process was allowed to finish. Only
its waiting controller was paused, and the idle matrix controller was replaced
under the same output lock. All heavy work stays serial in allocation 115927;
the priority controller restores the original pair and starts the direct-first
matrix after its bounded tests. No additional allocation or production change
is introduced. Compiler AO-buffer estimates remain distinct from measured RSS.

## 2026-11-05: explicit fresh-density RHF reference control

The original conventional PySCF RHF attempt exhausted its unchanged 200-main-
cycle budget. Its last energy changes stayed near 2.34e-9 Ha, while density
RMS and orbital gradient passed. It remains a failed reference and has no
accepted speed ratio. A separate control was specified in the task messages
and adapter before launch and documented in `reference_fresh_jk_plan.json`:
pass only `dm_last=0` and
`vhf_last=0` to the original bound `get_veff`, retaining `direct_scf=True`, its
screening optimizer, 1e-12 screening, initial guess and every stopping gate.

This tests incremental-product accumulation rather than simultaneously
disabling the optimizer. It neither overwrites the original attempt nor
increases the main-cycle budget. Three fake-object metadata tests were RED
before the adapter and then GREEN, including exception restoration and raw
record protection. The current diagnostic step continues uninterrupted; only
its waiting parent is paused until this single control finishes, then restored.

Chronology audit: the plan JSON's local timestamp is 18:04:45.915 +08:00,
while the remote controller logged launch at 18:04:39 +08:00. Therefore this
written file is not claimed as a pre-execution registration. The hypothesis,
adapter and unchanged gates were already specified; the initial JSON and
revision-9 lock are retained, and the corrected record states both timestamps.
