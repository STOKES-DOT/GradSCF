# NNAO compatibility and validation

The current native joint workflow learns contraction coefficients and shared
`(element,l)` log-exponent scales, `alpha = alpha_ref * exp(beta)`. The pool
workflow additionally has its existing role-specific scale/log-spacing
parameters. Model output interpretation, positive exponents, normalization,
fixed core, trainable masks, optimizer and checkpoint formats are preserved.
Graph positions use Angstrom; integral centers use bohr and exponents bohr^-2.

## Actual training requirement

The inspected training objective is a reconverged RHF energy mean. MACE is
called with force/stress output disabled. No active NNAO force-supervision
consumer was found. Its prior documentation already excludes primitive
basis/geometry mixed derivatives in the joint RI path. This migration does
not claim to add that missing capability. Generic existing force-loss products
and molecular geometry response have separate tests; they are not evidence
of a complete NNAO force-training chain.

The historical trained `best.npz` was unavailable. Validation uses the existing
MACE architecture with a declared small random fixture (seed3, one interaction,
four channels, H-only) at H2 distances0.74 and0.91 Angstrom. This tests an actual
model, implicit SCF, native pullback and optimizer, not trained-model accuracy.
The timing adapter assembles the small fixture instead of invoking the full
7171-state CHNOF campaign constructor, including its filesystem/source guards.
Import paths, source/model hashes and the actual native library are checked
separately. The numerical `NativeJointEngine.evaluate/molecule` and `JointAdam`
methods are unchanged. This does not validate a real campaign constructor or
resume of an unavailable trained checkpoint.

## Passed gates

`tests/nnao/test_libcint_consumers.py` covers:

1. Fixed-primitive projection versus contracted native inputs for the same
   physical basis at both geometries; integral parity2e-10.
2. Coefficient and actual log-exponent directions, finite differences at
   2e-4 and1e-4, atol2e-8/rtol2e-6; JVP/VJP inner products2e-10/2e-9.
3. Geometry and exponent changes invalidate fixed integral cache signatures.
4. qvszps C/H retains ECP/ncore/effective charges and coefficient response;
   unsupported contracted ECP projection raises before solving.
5. The real MACE fixture, complete reconverged energy parameter directions,
   finite differences, finite gradients, accepted optimizer state, fixed masks
   and checkpoint roundtrip. Complete energy FD gate2e-6 Ha/2e-5 relative.

Together with OFDFT representations and the bounded small-system coefficient
trajectory tests, `evidence/nnao-green.xml` records24 passed in141.81s.
This count is the selected test group, not the entire NNAO test suite.

`no-pyscf-smoke.log` independently exercises installed-package NNAO
coefficient/exponent forward and implicit SCF pullback, followed by an update,
without PySCF installed. Coefficient/exponent gradient norms are0.09901348 and
0.06069047. A loss decrease alone is not used as a correctness criterion.

## Timing and device boundary

`tools/libcint_migration/nnao_step.py` runs two separate fresh-process modes:
`full-step` for clean loss+gradient+Adam time and `components` for instrumented,
synchronized phase times. Each has one first step and three updated-parameter
warm steps plus an extra untimed validation/checkpoint roundtrip. Components
are not added to or subtracted from clean times to infer a backward cost.

| Small two-H2 fixture | First complete step | Median warm complete step | Process peak RSS |
|---|---:|---:|---:|
| CPU | 9.095s | 0.0191s | 1351.7MiB |
| A800 model/SCF + CPU libcint | 13.804s | 0.0585s | 1881.5MiB |

CPU and GPU loss trajectories agree within about2e-15 Ha in this fixture.
This small case favors CPU; it is not evidence for large-system GPU scaling.
GPU logs include hwloc binding warnings; the processes retain their recorded
Slurm CPU affinity. Integral projection and its pullback execute on CPU;
model/SCF/model VJP placement and transfers are explicit in JSON.

The JSON records radial/AO primitive counts, contracted AO count, auxiliary
count/rank, native and model hashes, cache behavior, all raw repeats and
component timing. RSS includes first compilation and all repeats, not a
separately isolated warm memory peak. Allocator reservation and actual device
usage must not be confused. These small true-RI runs do not form a full
primitive four-index ERI cache.

Raw records: `evidence/nnao-{cpu,gpu}-{full-step,components}.json` and logs.
No claim is made about a supplied real training dataset, trained weights,
independent-center head, or NNAO mixed force-loss derivatives.

The additional four metadata gates and the seven NNAO consumer tests pass
together (`cache-nnao-green.xml`:11 passed). Fixed caches now include the actual
library SHA256, normalization source, primitive centers/nuclei and auxiliary
parameters. Hash memoization has32 entries and is keyed by file stat identity.
Old integral cache keys explicitly require rebuilding; NPZ arrays and model /
optimizer checkpoint schemas are unchanged. Arbitrary caller-constructed
precomputed metric-factor matrices retain their existing signature/lindep
contract; no new matrix checksum policy is introduced here.

## Unified S9 acceptance

The n006/RTX5090 before/after CPU and GPU full-step runs all complete four
updates, final validation and checkpoint roundtrip. Offline inspection reads
the NPZ arrays and safely decodes optimizer state: count=4, frozen theta
indices [1,258] have zero gradients/moments, inactive beta entries stay zero.
CPU before/after parameter, gradient and optimizer arrays are bitwise equal;
GPU differences are approximately1e-15. This remains a random small-MACE
fixture, not the unavailable real campaign/checkpoint.

CPU1 warm full-step medians are0.018423→0.018165s; RTX5090/CPU8 medians are
0.050100→0.050470s. These do not establish a general training speedup. The
separate GPU component run records model/SCF/implicit response, native
pullback and transfer phases with extra synchronization; it is not added to
or subtracted from clean full-step timings. See `s9_results.json` and the
durable `evidence/s9/nnao-*.json` records.

The final installed package has PySCF unavailable, executes the representative
coefficient/exponent SCF pullback/update, and passes12 fresh-process geometric
JVP/VJP/HVP cases. All486 checked installed Python/C++/basis-resource files
match this worktree. The raw proof is `evidence/s9/installed-source-proof.json`.
