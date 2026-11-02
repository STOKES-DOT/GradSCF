# Conventional basis comparison on the ethane torsion scan

Identical rigid 25-point geometry grid, 0–120 degrees in 5-degree steps, DF-RHF, def2-universal-jkfit, float64, spherical AOs. NNAO uses frozen update 3351. Conventional basis coefficients and exponents are fixed at their published library values. NNAO free contraction is optimized independently for each geometry at fixed exponents and core basis.

| Basis | AO count | Grid barrier / kcal mol-1 | Maximum relative-curve difference from cc-pVTZ / kcal mol-1 |
| --- | ---: | ---: | ---: |
| STO-3G | 16 | 3.049505 | 0.166371 |
| 3-21G | 30 | 3.022156 | 0.193720 |
| 6-31G(d,p) | 58 | 3.223413 | 0.008373 |
| cc-pVDZ | 58 | 3.458704 | 0.242828 |
| cc-pVTZ | 144 | 3.215877 | 0.000000 |
| NNAO / MACE | 44 | 3.651135 | 0.435258 |
| NNAO / free contraction | 44 | 3.495155 | 0.279279 |

Each relative curve is referenced to its own 0-degree energy. HF/cc-pVTZ is the largest conventional basis in this comparison and is not an exact reference. The barriers refer to this rigid scan, with no nuclear-geometry relaxation.

Traditional basis calculations: GPU4PySCF, job 100380, one V100, 32 CPU, 54.24 seconds excluding Python imports. SCF energy tolerance 1e-12 Eh, orbital gradient tolerance 1e-9, maximum orbital residual < 1e-7. NNAO energies were independently checked on all 50 frozen endpoint bases.

Command: `sbatch --dependency=afterok:<dense-job-id> artifacts/nnao-ethane-torsion-20261018/traditional.slurm`.
Plots: `ethane-torsion-benchmark.png` and `.pdf`; all numerical curves: `benchmark-curves.csv`.
