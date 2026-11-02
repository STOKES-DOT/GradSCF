# Ethane torsion: MACE versus free contraction

Rigid 0–120 degree methyl rotation in 5 degree steps, starting from training geometry `GMTKN55-v2/ADIM6/AM2`; bond lengths and valence angles are fixed.
MACE uses the frozen update-3351 checkpoint. Exponents, 6-31G core contractions, single-zeta plus polarization topology and all-electron DF-RHF settings are identical.
All angles have 44 spherical AOs. Each free-contraction problem has independent per-atom radial coefficients, shared by the magnetic components of each shell; core coefficients stay fixed.
Two starts (MACE and reference coefficients), analytic implicit SCF gradients, L-BFGS-B, 1000-iteration cap; coefficient gradient norm <= 1e-5 is reported as gradient-converged. Finite-difference validation is not performed.
SCF tolerances: energy 1e-12 Eh, density 1e-10, orbital gradient 1e-9; implicit tolerance 1e-8. Native packed DF with def2-universal-jkfit; no full four-index ERI.

| Quantity | Value |
| --- | ---: |
| mace_grid_barrier_kcal_mol | 3.651135074 |
| free_grid_barrier_kcal_mol | 3.495155486 |
| barrier_difference_kcal_mol | 0.155979588 |
| absolute_gap_min_kcal_mol | 0.04195863812 |
| absolute_gap_max_kcal_mol | 0.1979382286 |
| relative_curve_max_abs_difference_kcal_mol | 0.155979588 |
| relative_curve_rms_difference_kcal_mol | 0.08807953634 |
| gradient_converged_points | 25 |
| gradient_converged_starts | 50 |
| maximum_start_spread_hartree | 1.622879608e-11 |
| maximum_independent_energy_error_hartree | 7.105427358e-13 |
| scan_seconds | 2629.719181 |
| independent_check_seconds | 45.36793677 |

Run on jy_gpu_2, Slurm job 100369, following coarse-grid job 100368; one V100, 32 CPU, 128 GiB requested RAM per job, run sequentially. Commands: `sbatch artifacts/nnao-ethane-torsion-20261018/run.slurm`, then `sbatch --dependency=afterok:<coarse-job-id> artifacts/nnao-ethane-torsion-20261018/dense.slurm`.
The 0–120 degree closure and mirror errors are retained in `metrics.json`. All 50 frozen bases are checked independently with GPU4PySCF. Finite-budget agreement between starts does not prove a global coefficient minimum. Grid barriers are measured on this rigid scan, not relaxed-geometry or complete-basis barriers.

Raw per-angle energies: `comparison.csv`; optimization histories and coefficients: `angle-*/`; full geometry sequence: `torsion-dense.xyz`.
