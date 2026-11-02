# Campaign status — 2026-10-29

## Physical-only formal execution

The user explicitly cancelled CP/BSSE and extra CPU/GPU comparisons.
Only the 30 frozen physical structures are calculated. Independent monomer
energies remain necessary for the uncorrected dimer interaction energies.
No ghost atoms or dimer-conditioned monomers are requested. Aggregation
neither reads CP outputs nor produces CP metrics/plots.

- Required: 1680 method-energy cases = 27 bases x 30 structures x 2 methods,
  plus the same 60 cases for the finite-reference basis.
- All 1320 physical results available at the scope change were preserved.
  The original numerical source is frozen byte-for-byte as run_gpu_engine.py;
  the new workflow preserves its energy identity and reuses valid results.
- Small/SZP/DZP: 960/960 physical cases complete. The old mixed-scope lane
  report, whose failures were CP-related, is archived under history/.
- TZ/TZP job 111962 resumed only missing physical cases; its final log shows
  the remaining 120 cases have completed. Read lane-tz.json for current status.
- QZ/reference job 111963: 240 cases remain; the resource controller releases
  it within the existing maximum three GPUs / 64 account CPUs limits.
- The existing scope-specific failures and CP data are preserved as history
  and excluded from the physical-only accuracy score. They do not trigger
  recomputation or repair in this campaign.
- Controller remains in --direct mode. preflight_passed is null, not a
  fabricated pass; CPU comparisons and CP/BSSE calculations are disabled.

## Runtime and fixed inputs

- Independent gpu4pyscf_bench: GPU4PySCF 1.7.0, PySCF 2.8.0, CuPy 13.4.1,
  cuTENSOR 2.2.0, NumPy 2.2.6, SciPy 1.15.3. Original jax_scf unchanged.
- Complete CUDA 12.8 Toolkit comes from cluster NVHPC 25.3. Conda hooks select
  its compiler/runtime/math libraries and preserve pinned pip cuTENSOR 2.2.0.
  Real V100 import, contraction and generalized eigensolver passed in 111899.
- 30 physical structures and all 28 orbital-basis files (27 compared + one
  reference) remain frozen, together with explicit auxiliary basis definitions.
  All nine checkpoint/source-matched model exports passed before this run.
- Final NNAO checkpoints are used; L-SZP is update 5969. No retraining or free
  contraction is performed in this campaign.
- CPU MACE inference previously passed. No new CPU/GPU inference or energy
  comparison is required to run the frozen exported bases.

## Paths and delivery

Remote: /home/user_yuanjiao/private/homefile/NNAO_basis_accuracy_20261029.
Raw physical energies, SCF logs and explicit failures remain under results/, scf-logs/
and failures/. Current formal calculations have not finished; no final MAE or
accuracy/size plot is available yet. Inspect live Slurm state and scheduler.json
before reporting progress. Preserve source-overlap labels and do not omit failed
cases from aggregate accuracy scores.
