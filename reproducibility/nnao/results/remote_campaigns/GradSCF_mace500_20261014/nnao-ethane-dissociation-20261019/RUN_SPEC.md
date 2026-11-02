# Ethane C-C stretch: MACE versus free contraction

This is the user-approved restricted DF-RHF pressure test, not a physically correct
homolytic dissociation curve. The two methyl fragments retain their original
pyramidal geometry, bond lengths, and internal angles. No geometry relaxation,
unrestricted promotion, or electronic-state substitution is performed.

- Frozen MACE update 3351, parameter SHA256
  `2f4386c12cb4637b0b1f92682c86dba2b298b1e9c27f1926f1c2e24a1f6f1ade`.
- Same 44 AO layout, calibrated exponent pool and fixed 6-31G core as the torsion
  experiment. C has one valence s, one valence p and one d polarization channel.
- DF-RHF/def2-universal-jkfit, spherical AOs, float64. Existing implicit SCF and
  stability checks unchanged. No full ERI and no finite differences.
- 56 distances: original 1.525976376756 Å; 1.55–3.00 Å in 0.05 Å steps;
  3.10–5.00 Å in 0.10 Å steps; 5.25, 5.5, 6, 7, 8 Å.
- At each new distance, MACE is evaluated and contractions are optimized from
  both MACE and reference coefficients using the existing L-BFGS-B protocol.
  All trainable contractions on all atoms are optimized; C0 s/p are inspected.
- Maximum 1000 iterations/start, energy tolerance 1e-12 Eh, density 1e-10,
  orbital gradient 1e-9, implicit tolerance 1e-8. Coefficient gradient norm
  <=1e-5 is the reported endpoint convergence criterion, not proof of globality.
- Baseline at the original geometry reuses the already checked torsion endpoint.
  Stretched geometries are outside the original single training geometry.
- Failed SCF/implicit gradients and unconverged optimization endpoints are retained
  separately. They must not be plotted as converged free-contraction results.
- Analysis: C0 valence R_s/R_p, radial moments, MACE-minus-free energy gap and
  distance-dependent curve changes. Align phases and use physical normalization.
- Prospective Slurm resources: one V100, 32 CPU, 128 GiB, maximum three hours.
  Existing training 99005 is not stopped or modified.

Prepare locally:
`/opt/anaconda3/bin/python artifacts/nnao-ethane-dissociation-20261019/prepare.py`.
Submit from the existing frozen remote checkout only after resource approval:
`sbatch artifacts/nnao-ethane-dissociation-20261019/run.slurm`.

The adapted point runner is generated from the previous experiment; the exact
source hash and geometry/metadata-only modifications are in `preparation.json`.
The frozen scientific source hash and checkpoint file hash are checked by each
point before numerical work. Source files in `src/` are not changed by this test.
