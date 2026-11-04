# Phenanthrene cation SCF convergence diagnosis

2026-11-02. Production baseline: 2043d7d; current worktree 709e4ea.
The only production change since that baseline affects the UKS TDDFT response,
after SCF, so it cannot explain the SCF failure.

## Evidence before rerunning

Neutral phenanthrene RKS/PBE converged in 82 cycles. The paired cation UKS/PBE
reached 200 cycles without passing the SCF gate. Neither process hit its RSS
limit. This does not establish a general inability to converge large molecules.

The cation used Cartesian def2-TZVP, grid3, weigend auxiliary RI, all electrons,
47 alpha / 46 beta occupations, the neutral PubChem 995 geometry, and an hcore
guess. Energy, AO density RMS and orbital-gradient thresholds were 1e-12 Ha,
1e-11 and 1e-10, respectively, required simultaneously. The orbital-gradient
norm is the orthonormal-basis Fock/density commutator norm divided by sqrt(2),
computed at the new density using the raw Fock, not the extrapolated Fock.

The failed public result selects the smallest density-step RMS iterate, not
necessarily the last iterate. Its finite energy alone establishes neither
stationarity nor a stable terminal sequence. The public result drops the
internal histories. The original failure cannot retrospectively distinguish
oscillation, slow progress, occupation changes or a near-convergence floor.

## Bounded diagnostic protocol

- [x] Audit the actual UKS convergence and DIIS source without changing it.
- [x] Rerun the exact cation input with scalar iteration telemetry and save the returned state.
- [x] Compare an independent PySCF run, with quadrature and convergence-policy differences explicit.
- [x] Explain which criterion fails and identify the next single-factor experiment, before production fixes.

Slurm 114641: one V100 (allocation only), 64 logical CPUs, 256 GiB, n003.
All scientific calculations use CPU float64. OH/def2-SVP is a short hook smoke
test before the large case. The instrumented GradSCF run retains the original
200 cycles and three thresholds. It logs absolute energy, delta energy,
density RMS, physical orbital residual, DIIS-Fock alpha/beta gaps, electron
counts, S squared, and the selected cycle. Runtime/source/native/input hashes
are saved. Callbacks are diagnostic overhead, so these times are not clean
performance measurements. Single calculation limit is 6000 seconds and RSS
guard 200 GiB. PySCF is separately bounded to 1800 seconds.

No production SCF kernel, threshold, initial guess or physical approximation
was changed. The PySCF comparator has its own grid3 quadrature and requires
energy/orbital convergence; it does not implement GradSCF's independent density
RMS threshold. Its convergence result must therefore be compared using the
recorded metrics, not its converged flag alone.

Probe: `benchmarks/probes/response_grid3/scf_diagnostic.py`.
Remote evidence: `gradscf-response-grid3-20261101/results/scf-diagnosis-114641/`.

## Completed measurements

Both jobs completed with Slurm ExitCode 0:0, verified through `scontrol`.
Job 114641 ran 10:03:44--10:50:18; dependent job 114669 ran
10:50:18--10:51:07 on 2026-11-02, UTC+08. They were strictly serial;
resources have been released. Slurm accounting storage is disabled, so job
metadata came from live `scontrol`, and memory from process-group sampling and
Linux `ru_maxrss`, not `sacct`.

Hardware: n003, Xeon Gold 6230, 64 allocated logical CPUs (host has 40 physical
cores / 80 logical CPUs), 256 GiB. The V100 was allocated but unused by these
CPU-only calculations. Python3.11, JAX0.10.0, NumPy2.4.4, PySCF2.13.0;
full actual paths, hashes and affinities are in the raw runtime records.

Runtime dimensions: 564 AO, 974 auxiliary functions, 295732 grid points,
47/46 occupied alpha/beta orbitals. Native SHA256:
`ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8`.
SCF source SHA256 matched the local worktree. No production source changed.

The diagnostic reproduced the old returned energy exactly:
**-538.7550781778976 Ha**, selected from cycle **157**, not cycle 200.
The final cycle energy was -538.7550781779504 Ha. Public SCF convergence
remained false after 200 cycles. Instrumented SCF wall time was2188.029s,
peak RSS9.70GB; these are diagnostic measurements, not before/after timing.

| Cycle | absolute delta E (Ha) | AO density RMS | reported orbital gradient |
|---:|---:|---:|---:|
|20|1.18e-8|1.29e-6|6.66e-4|
|30|6.14e-12|8.94e-9|2.65e-6|
|40|5.53e-11|8.56e-11|2.93e-8|
|50|1.50e-11|6.20e-12|5.80e-10|
|100|6.37e-12|6.33e-12|4.71e-10|
|200|1.96e-11|7.01e-12|5.19e-10|

Across cycles60--200, all141 density tests passed, only7 energy tests passed,
and **zero orbital-gradient tests passed**. The gradient ranged4.41e-10 to
6.70e-10, median5.31e-10. Median absolute delta E was2.14e-11Ha; the entire
energy span was8.12e-11Ha. This is a terminal numerical plateau, not continued
macroscopic density/energy oscillation. Electron counts remained47/46, S squared
settled at0.75471935, and both spin gaps stayed positive. These do not prove
electronic stability or that the determinant is the global ground state.

## Confirmed residual-evaluation problem

The certificate rebuilt the same integrals/grid and raw Fock at the saved
density, without an SCF update. Its energy differed from the archived value
by only -2.27e-13Ha. Two stationarity evaluations on that same state gave:

| Evaluation | joint spin gradient norm | passes unchanged1e-10 threshold? |
|---|---:|---|
|Production AO commutator / sqrt(2)|5.2411e-10|no|
|Independent occupied--virtual MO block|3.4713e-11|yes|

For real, integer-occupation, complete S-orthonormal orbitals, exact arithmetic
gives

\[
 g^2=\sum_\sigma\|C_{v\sigma}^{T}F_\sigma C_{o\sigma}\|_F^2
 =\tfrac12\sum_\sigma\|X^T(F_\sigma D_\sigma S-SD_\sigma F_\sigma)X\|_F^2,
 \qquad X=S^{-1/2}.
\]

This identity requires no occupied/virtual degeneracy regularization and no
change to the physical SCF problem. UKS currently forms the AO difference
before multiplying by X; RKS already uses an occupied--virtual residual
(`scf/rks.py:171`, `scf/uks.py:638`). The discrepancy exposes finite-precision
limits of that UKS evaluation at this basis conditioning:

- minimum S eigenvalue3.0624e-6, maximum26.9302, condition number8.7939e6;
- zero eigenvalues were clipped at the1e-10 cutoff;
- inverse-root metric error3.68e-10, complete MO metric errors about3.77e-10;
- per-spin density idempotency errors about1.5e-13; electron counts correct;
- an independent64-mantissa-bit contraction of the float64 state gave MO
  residuals2.4234e-11 and2.4551e-11, consistent with the ordinary MO evaluation.

Thus one specific rejection mechanism is confirmed: **the AO-based stationarity
evaluation rejects a fixed state whose occupied--virtual MO residual passes the
same registered threshold**. This supports cancellation/metric-rounding
amplification; it is not evidence that the molecule lacks an SCF solution.
It also does not certify the complete original convergence gate: delta energy
has a separately observed noise problem, and its integral/grid/linear-algebra
contributions have not been isolated. Cycle157 had absolute delta energy
4.2292e-11Ha, so it still fails the joint gate even if its gradient is evaluated
in the MO representation. The measured metric error also means the exact norm
identity is only approximate at this precision; this probe has not uniquely
separated AO subtraction error from inverse-root/metric rounding.

The hcore guess is an additional early-iteration weakness, not the late plateau
explanation: initial energy was -313.7366Ha and the first step increased it by
195.3242Ha. Only hcore/explicit-density guesses are currently implemented.
Early transient costs and terminal numerical rejection should be optimized
separately.

## Independent PySCF result and its boundary

PySCF2.13.0, same molecule/basis/auxiliary basis/grid level/hcore guess,
conv_tol1e-12 and conv_tol_grad1e-10, reported convergence at161 cycles.
Its last ordinary iteration had delta E4.55e-13Ha, MO gradient6.03e-11,
and density RMS1.37e-12. Returned energy was -538.7550781779814Ha, differing
from GradSCF's selected energy by8.38e-11Ha. Independent quadratures and
algorithm choices remain explicit; no speed ratio is accepted here.

The extra final PySCF diagonalization then reported delta E1.36e-12 and
gradient1.35e-10 (printed precision). Those exceed the original strict
thresholds, but PySCF's extra-cycle check uses10*energy_tol OR3*gradient_tol.
Its public converged flag therefore cannot be interpreted as proof that its
returned state satisfies GradSCF's three-part final gate. Both programs were
challenged by this precision target; PySCF nevertheless reached a qualifying
ordinary iteration. Source checks:
[PySCF SCF driver](https://pyscf.org/_modules/pyscf/scf/hf.html),
[initial-guess definitions](https://pyscf.org/user/scf.html#initial-guess).

## Next production work, not implemented in this diagnosis

1. Use the shared, correctly normalized occupied--virtual residual for UKS
   convergence; keep the1e-10 gate and verify fresh/restarted states and AD.
2. Isolate and reduce the delta-energy evaluation noise with unchanged grid,
   XC, RI precision and thresholds. A residual-only change does not resolve
   this second acceptance issue by itself.
3. Improve the hcore start and DIIS robustness only after the terminal
   precision issue is controlled; use single-factor comparisons, not arbitrary
   extra cycles or a wholesale solver replacement.

OH smoke and its independent certificate passed. The phenanthrene diagnosis
completed as an intentionally retained nonconverged result; its fixed-state
occupied--virtual MO residual passed the original tolerance while the production
commutator residual did not. No full SCF fix, geometry
gradient, neural-functional validation or electronic-stability acceptance is
claimed by this diagnostic-only work.

Machine-readable summary: `scf_summary.json`; complete scalar trace:
`scf_iterations.csv`; immutable raw JSONL/logs: `raw/scf-diagnosis-114641/`.
The saved10MB state NPZ remains at the recorded remote path, with its SHA256
in `gradscf/residual_certificate.json`.

To reproduce inside an authorized remote Slurm allocation, from the candidate
source root (set PYTHONPATH to its src), use a new output path:

```bash
python benchmarks/probes/response_grid3/scf_diagnostic.py \
  reports/response_grid3/20261101-2043d7d/configs/first_large_true_ri/phenanthrene-cation-uks-pbe-ri-tda5-cpu64.json \
  /path/to/new-output
python benchmarks/probes/response_grid3/scf_residual_certificate.py \
  reports/response_grid3/20261101-2043d7d/configs/first_large_true_ri/phenanthrene-cation-uks-pbe-ri-tda5-cpu64.json \
  /path/to/new-output/scf_state.npz /path/to/new-output/residual_certificate.json
```

The independent reference uses the first command with `--engine pyscf` and
another empty output directory. Runs require at least64 allocated logical CPUs
for this exact large input; do not execute these scientific tests locally.
