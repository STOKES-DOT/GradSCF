# Gas-phase anthracene: eight-state full TDDFT FC/HT absorption

The current calculation includes **FC/linear-HT corrections for S1 through S8**,
with individual bands and their absolute-intensity sum on one photon-energy axis. It uses GradSCF for the
SCF, native integrals, nuclear response and full TDDFT. PySCF checks were explicitly skipped by the user; existing diagnostic logs
are retained, with no accepted external-reference claim or production dependency.

The original one-band input remains `input.json`; the active eight-state input
is `input_eight_states.json`: neutral anthracene C14H10, PubChem CID8418
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
   coordinate. Signed cross-geometry AO transition-density overlaps and a joint Hungarian
   assignment track all eight targets, including dark states. Each baseline
   and displaced solve requests twelve candidates, retaining four guard roots
   above the eight target states. Every solve retains all occupied/virtual orbitals. Complete energy-gap and transition-dipole
   slopes are central finite differences, including orbital and X/Y response.
   A second step checks the union of each state's strongest HR and HT mode
   (at most 16 modes/32 additional points). All 66 modes are retained for every state.
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

Ground data and the canceled one-band run remain under
`reports/anthracene_ht/20261110-b3lyp-def2svp/raw/`. New eight-state outputs go
to `reports/anthracene_ht/20261110-b3lyp-def2svp/eight_states/`. The raw directory
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

## Eight-state execution and accepted boundaries

The source/native/geometry-identified ground structure, all72Hessian columns
and66positive modes are immutable inputs. The oldS1 points contain only one
aligned dipole, so their missing seven-state data cannot be recovered. New
points share one SCF/12-root full TDDFT solve, store every candidate C/X/Y/μ/E
and compare oldS1 signed observables after baseline phase alignment. They do
not perform eight independent SCF solves or rerun the ground Hessian.

Eight-state configuration/output selection uses two environment variables,
without a Python CLI:

```sh
export GRADSCF_ANTHRACENE_INPUT="$PWD/examples/tdscf/anthracene_ht/input_eight_states.json"
export GRADSCF_ANTHRACENE_OUTPUT="$PWD/reports/anthracene_ht/20261110-b3lyp-def2svp/eight_states/raw"
PYTHONPATH=src python examples/tdscf/anthracene_ht/03_mode_response.py
```

Run only inside the authorized allocation. Current queue23 checks36 regressions;
queue24 computes eight-state mode derivatives, queue25 plots, queue26 releases.
The old single-state process was stopped and its queued spectrum disabled.
No PySCF was resumed. Test receipts distinguish synthetic spectrum fixtures
from actual molecular derivatives, which remain running.

Per-state isolated-energy, overlap, margin and guard-boundary checks remain
strict; near-degenerate mixing is rejected rather than rotating distinct
energies to improve tracking. The model is still0K independent equal-curvature
VG with linearHT: no excited-state Hessians/Duschinsky, thermal populations or
inter-electronic-state vibronic Hamiltonian is introduced. Electronic-state
intensities are added incoherently on the common absolute energy axis; only
within each state is the signed FC×HT interference retained. Every band and
the sum preserve optical-area gates, without per-band peak normalization.
