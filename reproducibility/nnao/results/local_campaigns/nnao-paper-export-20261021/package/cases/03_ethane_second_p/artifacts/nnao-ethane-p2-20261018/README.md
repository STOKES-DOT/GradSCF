# Ethane: second carbon valence-p radial channel

Only the C valence-p radial multiplicity changes: [2s1p1d] -> [2s2p1d]. H remains [1s1p]. The compact AO count increases from 44 to 50; the mixed primitive pool remains 152. All exponents, fixed core basis functions, geometry, DF approximation and SCF tolerances are held fixed.
The added curve is free contraction only; the existing 44-AO MACE checkpoint is unchanged.

| Quantity | Value |
| --- | ---: |
| energy_lowering_0_kcal_mol | 2.346438061 |
| energy_lowering_60_kcal_mol | 2.708231469 |
| old_barrier_kcal_mol | 3.495155486 |
| new_barrier_kcal_mol | 3.133362078 |
| barrier_change_kcal_mol | -0.3617934076 |
| ccpvtz_barrier_kcal_mol | 3.215876615 |
| old_max_curve_difference_vs_ccpvtz_kcal_mol | 0.2792788711 |
| new_max_curve_difference_vs_ccpvtz_kcal_mol | 0.08251453655 |
| gradient_converged_points | 25 |
| independently_optimized_points | 15 |
| symmetry_transferred_checked_points | 10 |
| gradient_converged_starts | 30 |
| maximum_start_spread_hartree | 9.495693121e-11 |
| maximum_independent_energy_error_hartree | 9.947598301e-13 |
| elapsed_seconds | 3370.333988 |

RHF energies use native cached S/h/packed DF and existing implicit SCF gradients. The first half-period and any already completed second-half points use independent two-start optimization. Unfinished equivalent angles reuse coefficients through an exact reflected atom map, followed by a new SCF energy and full coefficient gradient evaluation at the target geometry. No curve interpolation is used. All 25 final frozen bases were independently checked with GPU4PySCF. Finite-budget optimizations from two starts do not prove global minima. HF/cc-pVTZ is a comparison basis, not an exact reference. All barriers are from the rigid scan.

Slurm jobs 100401, 100454, Tesla V100-SXM2-32GB, 32 CPU; total runtime 3370.33 seconds. No finite-difference validation was performed. See RUN_SPEC.md and configuration.json for settings.

Plots: `ethane-p2-relative.png/.pdf`, `ethane-p2-absolute.png/.pdf`. Raw curves: `curves.csv`; per-angle checkpoints and histories: `angle-*/`.
