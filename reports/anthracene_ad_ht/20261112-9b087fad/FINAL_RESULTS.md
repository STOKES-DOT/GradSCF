# Accepted anthracene AD-VG/linear-HT results

Verified2026-11-12 22:28+08. Worktree `/Users/jiaoyuan/.codex/worktrees/bd0a/GradSCF`,
branch `examples/anthracene-ht-20261110`. Production pullback repair commit
`3eae6eca`; source Python tree SHA `6564351da56ee361c433bfbc15698a2c97fe078ae1e9b9b8d82cf93d7f4b8a1b`.
Native SHA `9f8da65e0e55029dbfcf137eeefd328b5e947015ac0c40a4ae468c3810787e89`.

The actual eight-state run is accepted. Both `mode_response_status.json` and
`spectrum_status.json` are complete and share signature
`d9e016d19f59ace234112a0d9b75a9b38c2c82c9fcefbb58a3790d70bb71d9bc`.
All32 Cartesian output gradients were produced by AD and projected onto66
normal modes. Archived displaced GradSCF results validated four directions;
they did not supply a model derivative. PySCF was skipped as requested.

## Physical scope

Optimized gas-phase anthracene C14H10,24 atoms,94 electrons; B3LYP,Cartesian
def2-SVP,grid3,float64,260 AOs,47 occupied and213 virtual orbitals. Every
orbital is retained. The8 target singlets use one external guard root in
full-TDDFT; the guard is excluded from the spectrum. The previously accepted
72-column nuclear Hessian and66 positive normal modes are reused.

The spectrum is0K independent equal-curvature VG plus linear HT. Gaussian
FWHM100cm^-1 is display broadening. No thermal population,Duschinsky rotation,
excited-state Hessian,interstate vibronic Hamiltonian,empirical energy shift
or experimental agreement claim is included. Listed0-0 energies are model
origins,not excited-state geometry optimizations or experimental measurements.

## Completed gates

| Gate | Actual result |
|---|---:|
| Finite Cartesian gradients | 32/32,each24x3 |
| Projected gap/dipole derivatives | 8x66 / 8x3x66 |
| Independent signed FD modes | 1,18,14,16; all passed |
| Maximum FD gap error | 6.8609e-10 Ha/Q |
| Maximum FD dipole error | 1.0382e-7 e*bohr/Q |
| Cartesian gap translation residual | 1.5363e-13 Ha/bohr |
| Cartesian dipole translation residual | 2.5890e-11 e |
| Maximum per-state optical moment area error | 6.9040e-15 |
| Common-axis summed optical area error | 3.7275e-15 |

Q is measured in bohr*sqrt(electron mass). FD gap/dipole acceptance used
atol3e-8/3e-7 with rtol3e-3; translation limit1e-5; final area limit1e-5.
The optical moment checks concern this finite-state harmonic model,not the
full electronic Thomas-Reiche-Kuhn sum over all excited states.

## Cost and resources

One NV5090,64 logical CPU,256GiB on n006,job122200. Native geometry AD and its
response execute on CPU. The small separate GPU-array tests do not establish
GPU native-integral geometry AD. The actual AD process took31993.843s
(8h53min14s),including startup,compilation,shared VJP and32 adjoints/validation.
Its0.2s process-group RSS sampled peak was176860991488bytes(164.7146GiB),
below220GiB guard. `ru_maxrss` independently reported the same byte value.
Shared VJP forward/linearization:2412.948s. Spectrum process:6.301s. Controller
stages ended22:12:35 and22:12:50 with exit0; job absent from queue22:28:08.
Slurm accounting storage is disabled,so no historical sacct status is claimed.
The old112.57GB reverse-constant capture failure and diagnostic assertion
corrections remain documented in `implementation_report.md` and raw receipts.

## State-resolved results

| State | Vertical eV | Model0-0 eV | Integrated FC | Integrated HT | Integrated total |
|---|---:|---:|---:|---:|---:|
| S1 | 3.236108 | 3.017221 | 0.05935974 | 0.01339307 | 0.072417172 |
| S2 | 3.861543 | 3.774051 | 0.00094950459 | 0.00047780314 | 0.0014367776 |
| S3 | 4.534307 | 4.332648 | 3.5378075e-11 | 0.001169152 | 0.001169152 |
| S4 | 4.917852 | 4.757104 | 1.0167445e-10 | 0.14702798 | 0.14702798 |
| S5 | 5.245276 | 5.194170 | 1.9456731 | 0.008371587 | 1.9532403 |
| S6 | 5.427458 | 5.212865 | 3.3190393e-06 | 0.009340838 | 0.0093433954 |
| S7 | 5.531393 | 5.462833 | 9.9681399e-11 | 0.023367571 | 0.023367572 |
| S8 | 5.815372 | 5.635852 | 0.079230898 | 0.094423419 | 0.17229621 |

Each total includes its signed FC/HT interference. The electronic-state
intensities are summed incoherently without independent peak normalization.
S5 dominates the absolute total; the state panels retain the smaller bands.
Several dark Condon transitions acquire intensity in the linear-HT model.
These are predictions of this approximation,not external validation claims.

## Artifacts and reproduction

- [Total spectrum](../../anthracene_ht/20261110-b3lyp-def2svp/ad_eight_states/anthracene_fc_ht.png)
- [Eight state panels](../../anthracene_ht/20261110-b3lyp-def2svp/ad_eight_states/anthracene_fc_ht_states.png)
- [Huang-Rhys map](../../anthracene_ht/20261110-b3lyp-def2svp/ad_eight_states/anthracene_mode_huang_rhys.png)
- [Vertical target sticks](../../anthracene_ht/20261110-b3lyp-def2svp/ad_eight_states/anthracene_vertical.png)
- [Band table](../../anthracene_ht/20261110-b3lyp-def2svp/ad_eight_states/bands.csv)
- [Full spectrum CSV](../../anthracene_ht/20261110-b3lyp-def2svp/ad_eight_states/spectrum.csv)
- [Mode table](../../anthracene_ht/20261110-b3lyp-def2svp/ad_eight_states/modes.csv)

PDF counterparts are beside the PNGs. `raw/` contains the32 checkpoints,
Cartesian Jacobian,projected derivatives,spectrum NPZ,and complete statuses.
All four PNGs were actually opened for visual inspection. The selected
figures/tables and hashes are versionable; large raw arrays and full CSV
remain local artifacts. `result_files.json` records artifact hashes.
Reproduce inside an authorized allocation with the example environment
variables and05/04 commands in `commands.txt`; the restart Slurm script
preserves success gates and hashes for reuse of its accepted B3LYP check.
No local scientific calculation,test,additional remote job or production
edit was needed for final artifact review. Original workspace is untouched.
