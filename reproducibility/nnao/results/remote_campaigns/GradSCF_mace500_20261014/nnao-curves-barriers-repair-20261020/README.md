# Native SCF repair of the representative basis benchmark

## Scope and status

Original report: `artifacts/nnao-curves-barriers-20261020/representative/`.
Its SHA256 is `53601e9a2abbb92251b942d5cd7e6fb9110664c21be1e480be0e2b15cff3cd94`.
It remains unchanged. Job **102347** was submitted using the previously
approved single A100 80G / 32 CPU / 512 GiB / three-hour limit. At the
2026-10-20 23:41 +08:00 check it was pending resources, not yet validated remotely.

The explicit retry selection is **242 cases**: all 21 failed NCI cases,
17 other points in the affected basis/curve combinations, and all 204 BH9
species. The latter makes the branch-selection protocol consistent before
reaction energies are rebuilt. Untouched cases retain their original provenance.

## Diagnosis

Six same-shell local CPU diagnostics are recorded in `audit-local.json`.
Native overlap, hcore and density-fitted J/K agree with independent PySCF.
The largest fixed-density energy difference is **7.73e-12 Eh**. No full
four-index ERI is retained in either diagnostic path.

All three representative failed dimers converge in the independent reference,
whereas the old core-Hamiltonian start and single small-shift rescue fail.
The native solver also reaches the reference energy when supplied that density
as a diagnostic. Reference densities are not used by the native retry workflow.

Two BH9 examples that passed the old stationarity test are much higher-energy
SCF branches: NNAO C4H7 product and 6-31G(d,p) C4H7 reactant. The larger
cc-pVTZ halogen-transfer TS also has a lower native stationary solution;
its independent cold-start SCF did not fully converge during the initial audit,
so its last iterate is not labelled a converged reference.

## Minimal repair

- Project bundled MINAO neutral atomic densities into the exact current
  orbital basis using native overlap integrals. Ghosts contribute no reference
  electrons. Normalize the initial alpha/beta traces to the declared charge/spin.
  This changes the initial guess, not the Hamiltonian or basis parameters.
- Retain the existing RKS/UKS DIIS kernels and strict convergence tolerances.
  If required, restart from the atomic seed with bounded level shifts and
  accept only a subsequently converged zero-shift solution.
- Check real internal orbital curvature with exact DF Hessian actions and
  the shared eigensolver. Restricted calculations stay restricted. UHF keeps
  its declared alpha/beta populations. Follow resolved negative modes with
  bounded native restarts; unresolved checks remain failures.
- For ghost calculations, also try the native own-fragment density embedded
  in the ghost basis. Preserve the common dimer auxiliary metric and nested
  variational-space checks.

The DF Hessian-vector action matches independent JAX autodiff on small
spin-coupled systems. No finite-difference derivative checks are performed.
Resolved lowest Ritz roots check local real internal stability; they do not
certify a global HF minimum or RHF-to-UHF / UHF-to-GHF stability.

## Verification and output

Local checks include native-only recovery of HBr-ammonia/cc-pVDZ and the
C4H7 reactant/6-31G(d,p) low-energy branch against independently computed
energies, atomic spin/charge traces, ghost CP invariants and Hessian tests.
The final combined native/core/protocol/aggregation run passed **38 tests**
in 26.76 seconds. Job 102347 repeats focused checks before numerical retries.

Runtime output is saved after each case under `retry/`. A failed priority
diagnostic stops the broad retry instead of consuming the rest of the budget.
Separate `merged/` outputs rebuild the 852-case table and the 54 BH9 profiles;
failed revalidations remain explicit. Plots are regenerated from accepted
retries, and `retry/independent-check.json` independently checks all repaired
original failures plus the three BH9 outliers. Six priority cases also receive
independent internal-stability checks. PySCF is confined to those comparison
scripts; native re-evaluation does not depend on its calculations.

Source changes are confined to comparison helpers and their tests. Production
SCF/integral sources and checkpoint update 3351 are unchanged. `native-before.py`
preserves the original comparison adapter; its hash matches the parent run.

Main commands are in `retry.slurm`. No new training, coefficient fitting,
exponent updates, or finite-difference validation is part of this repair.
