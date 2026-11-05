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
