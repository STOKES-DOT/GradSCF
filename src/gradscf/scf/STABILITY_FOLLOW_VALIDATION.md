# Spin-channel and negative-mode following validation

2026-09-23. Baseline e3a3e0a. macOS arm64 CPU (`TFRT_CPU_0`), Python 3.12.2,
JAX 0.8.1, PySCF 2.9.0, float64, OMP_NUM_THREADS=1. Timings include compilation
and are not performance benchmarks. No global SCF minimum or GPU claim is made.

## Channel distinction and independent oracle

Restricted internal coordinates rotate both spin orbital sets together. The
new spin channel rotates them oppositely: (+theta,-theta). Both use the same
single-spin angle scale. The spin energy is the original unrestricted evaluator;
its full gradient is checked before curvature classification. This prevents a
zero spin-projected gradient from hiding an unconverged charge state.

H2 at 2.5 Angstrom, STO-3G, native RHF:

- restricted internal minimum curvature: +2.471738603679 Ha;
- opposite-spin minimum curvature: -2.043622131927 Ha.

This reference is internally stable but spin-unstable. H2 at 0.74 Angstrom is
stable in both tested real channels. H2 and H2O spin-channel curvatures were
compared against four times PySCF's analytic RHF->UHF response action, with
absolute test tolerance 2e-9 Ha per squared angle. This is the coordinate
normalization factor, not a fitted scaling. The oracle is test-only; production
uses shared JAX energies, HVPs and `solve_hermitian`.

## Native directed restarts

The no-CLI example `examples/cc/stability_following.py` uses STO-3G, SCF
energy/density/gradient tolerances 1e-12/1e-10/1e-9, max_cycle=250. It tries
step_sizes=(0.5,1.0), both signs, max_restarts=5, energy acceptance threshold
1e-10 Ha. Initial channels use a dense Hessian for reporting; following uses
the default bounded Davidson curvature checks.

| Fixture | Requested channel | Initial energy / Ha | Final energy / Ha | Accepted rounds | Final model |
| --- | --- | ---: | ---: | ---: | --- |
| H2, 2.5 A | spin | -0.702943599724 | -0.933867203133 | 1 | UKS with HF functional |
| N2, 1.1 A | internal | -106.769673857813 | -107.496500511798 | 1 | RKS with HF functional |
| F2, 2.2 A | internal | -195.524616699278 | -195.671626240736 | 2 | RKS with HF functional |

F2 passes through -195.596566567657 Ha before reaching the lower branch.
Final lowest curvatures are 1.119927718131, 1.082138247105 and 0.151716184571 Ha,
respectively, in each final model's own coordinate space. All histories strictly
decrease; no source object is changed. The complete example took 22.67 s,
including compilation. Positive Davidson values remain local Ritz evidence,
with `spectrum_certified=False`; they are not complete-spectrum certificates.

The H2 unrestricted energy additionally matches an independently reconverged
PySCF UHF reference at 1e-9 Ha tolerance. Entering UKS is explicit and occurs
only after an accepted spin-breaking candidate. The existing closed-shell EOM
interface rejects the resulting unrestricted reference rather than silently
changing its method.

## Failure and freshness checks

Tests cover zero restart budget, unsupported channels, invalid step controls,
nonlower trials, both signed steps, monotone history and acceptance counts.
An existing fresh stable state is analyzed without rerunning SCF, even with a
zero budget. Stale public orbitals, configuration or energy are rejected for
both RKS and UKS. Trial-step iterables are normalized once before reuse.

Independent review reproduced a stale-energy defect before its fix: replacing
H2's public e_tot by e_tot-10 caused physically lower UHF candidates to be
rejected. The shared snapshot validator now checks that public e_tot equals
the stored RKS total_energy or UKS mf_energy. A public regression failed before
the fix and checks both restricted and unrestricted paths afterward. This
validator also removes duplicated R/U freshness logic.

## Final checks

The selected regression completed **38 passed, 3 skipped in 132.17 s** after
the freshness repair. It covers new following/channel behavior, original
R/U stability, precision/multistart, orbital optimization and orbital-gradient
tests. The skipped tests require optional `jax_xc`. The multi-dimensional water
spin-response oracle also passed independently before the combined run.
Independent read-only review and the final patch review found no remaining
blocking issue. This is not a complete repository test-suite claim.

The channel distinction follows the standard
[SCF internal/external stability convention](https://pyscf.org/user/scf.html#stability-analysis).
PySCF 2.9.0 supplies an independent analytic test action; no upstream numerical
iteration code was incorporated in the production workflow.

## Reproduction

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python examples/cc/stability_following.py

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_scf_stability_following.py tests/test_scf_precision.py \
  tests/test_restricted_multistart.py tests/test_uhf_stability.py \
  tests/test_uks_stability.py tests/test_scf_orbital_optimization.py \
  tests/test_scf_orbital_gradients.py
```

Only HF channel comparisons ran here. Optional DFT XC stability cases require
`jax_xc`, unavailable in this environment. Complex orbitals, spin-flip GHF,
ROHF mode following, derivative propagation through discrete branch selection,
and global-minimum guarantees are outside this increment. Conventional
single-run SCF and post-HF defaults are preserved.
