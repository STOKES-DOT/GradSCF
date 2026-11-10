# Gas-phase anthracene: full TDDFT FC/HT absorption

This example targets the **lowest isolated singlet vibronic absorption band**,
with a separate eight-singlet vertical stick spectrum. It uses GradSCF for the
SCF, native integrals, nuclear response and full TDDFT. PySCF checks were explicitly skipped by the user; existing diagnostic logs
are retained, with no accepted external-reference claim or production dependency.

Inputs are fixed in `input.json`: neutral anthracene C14H10, PubChem CID8418
conformer as an initial guess, Cartesian def2-SVP, B3LYP (including VWN-RPA),
grid level3, float64. Geometry optimization requires max|gradient|<=3e-6
Ha/bohr. The allocated resource is one NV5090/64 logical CPU/256GiB, job119234
on jy_gpu_2. Run scripts from the repository root inside that allocation;
there is no Python CLI or local scientific testing.

The stages are:

1. `00_initial.py`: actual SCF/full-TDDFT capability, dimensions and cost at
   the initial conformer. This is explicitly **not** an equilibrium spectrum.
2. `compare_initial_pyscf.py` is an optional diagnostic that was canceled by
   the user. The active pipeline has no PySCF prerequisite or further PySCF jobs.
3. `01_optimize.py`: fresh SCF at each geometry, then the common
   `MolecularEnergy(forward="reference")` implicit nuclear force. The current
   converged orbitals are dynamic inputs; no orbital-minimization loop is
   needlessly repeated to obtain the derivative.
4. `02_harmonics.py`: complete JVP-of-gradient Cartesian Hessian, with 72
   individually checkpointed columns, raw symmetry/translation checks and
   all 66 positive internal modes. No imaginary frequency is clipped.
5. `03_mode_response.py`: recompute SCF and full TDDFT at each +/- normal
   coordinate. Cross-geometry AO transition-density overlaps identify S1;
   signed overlaps align its phase. Each displaced solve requests four low
   roots to track the isolated S1; the baseline retains eight vertical sticks,
   and every solve retains all occupied/virtual orbitals. Complete energy-gap and transition-dipole
   slopes are central finite differences, including orbital and X/Y response.
   A second step checks important HR and HT modes. All 66 modes are retained.
6. `04_spectrum.py`: call the tested `gradscf_tools.vibronic` oscillator
   product and plot FC, HT, signed interference and the coherent total.

The existing default full-TDDFT facade attaches AD only to excitation energies
and stops X/Y amplitude derivatives. Consequently `jacrev(transition_dipole)`
would be incomplete here. This example explicitly uses **FD-HT**, not such an
incomplete AD derivative. Ground-state gradients/Hessian remain implicit AD.

## Model and units

The first spectrum is a **0 K, equal-curvature vertical-gradient (VG) harmonic
model**. It uses ground normal coordinates Q in bohr*sqrt(electron mass),
frequencies in Ha, g=d(Ee-Eg)/dQ and kappa=dmu/dQ. The excited surface has
its minimum shifted by -g/omega² with unchanged curvature and J=I:

`E00_model = E_vertical - sum(g²/(2*omega²))`.

This is not an optimized excited-state adiabatic-Hessian/Duschinsky spectrum.
Thermal populations, nonadiabatic coupling and lifetime broadening are excluded.
The Gaussian FWHM100cm^-1 is an explicit display convolution. There is no
empirical energy alignment, frequency scaling or separate normalization of FC,
HT and interference. The analytic time correlation scales with modes and time
samples, avoiding an exponentially large Fock-state enumeration.

Mode displacements returned by `normal_modes` use electron-mass atomic units.
Actual FD steps are specified in bohr*sqrt(amu); the derivative conversion by
sqrt(AMU_TO_ELECTRON_MASS) occurs exactly once. The saved vertical anchor fixes
the exact C/X/Y/dipole gauge. Source/native/input fingerprints protect all
checkpoints; incompatible source or parameters require a fresh run directory.

## Outputs and evidence

Outputs go to `reports/anthracene_ht/20261110-b3lyp-def2svp/`. The raw directory
contains native/source provenance, geometries, Hessian columns, signed mode
points, numerical gates and per-stage progress. A running or completed
preliminary SCF is not a completed vibronic spectrum. The final state is only
accepted when `raw/spectrum_status.json` has `complete=true` after all upstream
gates. Raw arrays and build/cache files remain outside Git.

The final deliverables are `anthracene_fc_ht.png/.pdf`, vertical stick plots,
`spectrum.csv`, `modes.csv` and source-identified numerical records. In an
existing allocated environment, a stage command is simply:

```sh
PYTHONPATH=src python examples/tdscf/anthracene_ht/01_optimize.py
```

The saved Slurm controller runs stages serially with prerequisite checks,
process memory guards and durable checkpoints. It uses at most the single
allocation; it does not submit parallel derivative jobs outside the allocation.

## Theory references and comparison limits

Dierksen and Grimme, [JCP120,3544 (2004)](https://doi.org/10.1063/1.1642595),
provide the TDDFT/B3LYP FC/HT context including anthracene. Their complete
vibronic models should not be equated with this cheaper VG approximation.
Benkyi etal., [PCCP21,21094 (2019)](https://doi.org/10.1039/C9CP04178H),
consider related PAH spectra with different functional/basis and Duschinsky.
For isolated-molecule context, [Bejaoui etal., JPCA127,2717 (2023)](https://doi.org/10.1021/acs.jpca.2c08994)
report a gas-jet anthracene band. Experimental 0-0 origins and theoretical
vertical gaps are distinct quantities. No experimental curve or agreement
claim is supplied until explicitly matched data and environments are available.

The first optimization process reported a nonfatal persistent-cache atime write
warning from JAX's disk cache. Numerical force/FD gates still passed. Later
stages disable the persistent disk cache and retain ordinary in-process program
reuse, avoiding cache-file eviction/I/O coupling; physical inputs are unchanged.
