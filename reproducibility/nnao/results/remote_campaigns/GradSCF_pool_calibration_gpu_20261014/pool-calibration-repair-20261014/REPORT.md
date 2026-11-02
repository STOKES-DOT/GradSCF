# S-DZ / S-TZ calibration repair

## Reproduced cause

The failed production jobs 96741 and 96744 ran A/B entirely on CPU, although
their allocations included a GPU. CPU reproducer 97296 (32 CPUs, 64 GiB,
n002, float64) completed 25 references, then aborted while compiling Kr2.
External /proc sampling captured exactly 65,530 mappings, equal to the node's
vm.max_map_count. Peak RSS was 12.14 GiB: this was a mapping-limit failure,
not exhaustion of the requested 256 GiB host memory.

## Controlled comparison

Prototype 97305 (A800 on n005, 32 CPUs, 32 GiB) retained CPU native integrals
and exponent VJPs but placed SCF and implicit response on GPU. It completed
all 26 references; peak mappings were 30,744 and peak RSS 10.33 GiB.
Across the 25 common completed references, maximum energy difference was
4.55e-12 Eh and maximum gradient-norm difference 3.52e-11. Gradient norms
alone do not establish equality of every gradient component; explicit
component comparisons and finite differences are included in the GPU tests.
These are cold-call capacity observations, not hot-training speed benchmarks.

## Implementation

PoolRHF accepts an explicit SCF device. Native plans, auxiliary metrics,
integrals and exponent pullbacks stay on CPU; physical arrays and their
cotangents cross the CPU/GPU boundary explicitly. Stage B keeps its fixed
integrals on the selected device. The default FreeContractionRHF and
minimize_reference implementations are AST-identical to the parent, so
fixed-exponent MACE and optimizer semantics are unchanged.

The standalone pilot defaults to CPU and accepts --scf-platform cuda.
The GPU basis-grid workflow defaults to --calibration-platform cuda.
Neither branch relaxes SCF/implicit tolerances or silently falls back.
Progress and completed timings are written even if a later phase fails.

## Validation and restart

Local focused regressions passed; GPU-specific pytest checks are skipped
without an allocated CUDA device. Complete validation job 97314 requests
one A100 40GB, 32 CPUs, 512 GiB and four hours. It runs CPU/GPU regressions,
one actual exponent-update/free-contraction step on all 26 references for
S-DZ and S-TZ, then replays all nine accepted fixed-exponent checkpoints.
The original raw parameters, gradients and Adam bytes are preserved.

Only after every gate passes does deploy.py publish the three validated
Python files and new nine-family gate to the idle queued campaign checkout.
It refuses unexpected source changes or an active job in that checkout.
Original 200-system job 96984 uses a different checkout and is untouched.

New production heads:

| Family | Previous failed job | Restart job | Resources |
|---|---:|---:|---|
| S-DZ | 96741 | 97318 | A100 80GB, 32 CPU, 512 GiB, 72 h |
| S-TZ | 96744 | 97319 | A100 80GB, 32 CPU, 512 GiB, 72 h |

Both depend on successful validation 97314. Jobs 96742 and 96745 now depend
on the new heads. SZ job 96738 waits for both 96984 and validation 97314;
other lane dependencies are retained. The last observed state was queued,
not completed calibration or production training.
