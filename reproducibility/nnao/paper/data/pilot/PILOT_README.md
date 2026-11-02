# GMTKN55 NNAO small-sample RHF validation

This file is an English translation of the imported source record, not a new run or result. Dates, status statements, and numerical values refer to that source record.

**The validation chain passed 23/24; overall admission failed.**
The original run recorded 23 passes and 1 failure; 0 explicit rerun records were substituted. The original failure remains in `verification.json`.

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for execution history, provenance exclusions, and the unresolved issue. The formal solver was restored to the measurement baseline.

## Actual method and scope

- GMTKN55 v2: 24 CHNOF, neutral, spin=0 records; names and ordering come from the saved `systems.csv`.
- RHF, `szp663_direct/core6`, spherical AOs; geometry, primitive exponents, and inner-shell parameters were fixed, while MACE predicted contractions for the learnable shells.
- CPU native evaluation generated fixed primitive overlap, hcore, and packed DF factors; GPU/JAX performed contraction, JK, SCF, and implicit backward. No full four-index ERI was generated.
- DF auxiliary basis `def2-universal-jkfit`, Adam learning rate `0.0001`, seed `0`, and 5 steps per system.
- Main-run SCF configuration: `{"conv_tol": 1e-12, "conv_tol_density": 1e-10, "conv_tol_grad": 1e-09, "max_cycle": 150}`; implicit configuration: `{"max_iter": 100, "restart": 40, "tolerance": 1e-09}`. The individual configurations for replacement records are stored in their source snapshots.
- Each structure independently initialized and optimized its own MACE parameters with a common CHNOF vocabulary; this was not shared-model training and does not establish cross-system generalization.
- Initial contraction outputs and initial MACE parameters were checked with fourth-order central finite differences; every perturbation reconverged SCF.
- FD step `0.0001`; acceptance was `atol=2e-06` plus `rtol=2e-05` times the absolute reference derivative.

Hardware/software record: n002 / Tesla V100-SXM2-32GB / JAX 0.10.0 / float64.

## Results

- Maximum initial-output directional AD–FD absolute error: **3.884e-09 Ha per unit direction**.
- Maximum initial-MACE-parameter directional AD–FD absolute error: **4.070e-09 Ha per unit direction**.
- Energy-decrease range over 5 steps in passing records: **0.128592–9.767895 mHa**.
- Maximum orbital residual among passing records: `1.459e-09`; minimum overlap eigenvalue: `4.757e-02`; maximum energy-reconstruction difference: `1.137e-13` Ha.

| ID | System | AO / primitive AO | Initial E / Ha | Final E / Ha | Energy decrease / mHa | Output FD error | Parameter FD error | Adam steps | Validation seconds | Outcome |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W4-11/h2 | Hydrogen | 8 / 24 | -1.125495609577 | -1.125624201287 | 0.128592 | 1.638e-11 | 3.787e-12 | 5 | 23.25 | Passed |
| W4-11/ch4 | Methane | 26 / 93 | -40.159891255462 | -40.160702453000 | 0.811198 | 1.996e-10 | 6.091e-10 | 5 | 25.86 | Passed |
| W4-11/nh3 | Ammonia | 22 / 81 | -56.087155405056 | -56.088337472657 | 1.182068 | 3.760e-10 | 2.398e-10 | 5 | 25.63 | Passed |
| W4-11/h2o | Water | 18 / 69 | -75.812708387912 | -75.814869345758 | 2.160958 | 1.480e-10 | 1.739e-10 | 5 | 24.20 | Passed |
| W4-11/hf | Hydrogen fluoride | 14 / 57 | -99.643752240916 | -99.647720389052 | 3.968148 | 4.773e-10 | 4.070e-09 | 5 | 20.79 | Passed |
| W4-11/co | Carbon monoxide | 20 / 90 | -112.413222923536 | -112.415837778127 | 2.614855 | 1.501e-10 | 1.210e-10 | 5 | 21.88 | Passed |
| W4-11/n2 | Nitrogen | 20 / 90 | -107.955458999093 | -107.957798103130 | 2.339104 | 3.884e-09 | 6.881e-10 | 5 | 21.09 | Passed |
| W4-11/c2h6 | Ethane | 44 / 162 | -79.152600310378 | -79.153990444351 | 1.390134 | 5.663e-10 | 1.018e-09 | 5 | 35.34 | Passed |
| W4-11/c2h4 | Ethene | 36 / 138 | -77.945388691232 | -77.946533692904 | 1.145002 | 5.447e-10 | 5.745e-10 | 5 | 29.30 | Passed |
| W4-11/c2h2 | Ethyne | 28 / 114 | -76.737882141942 | -76.738911782586 | 1.029641 | 6.699e-10 | 1.594e-10 | 5 | 25.51 | Passed |
| W4-11/methanol | Methanol | 36 / 138 | -114.779294899530 | -114.782127151048 | 2.832252 | 2.038e-09 | 7.633e-10 | 5 | 28.64 | Passed |
| W4-11/ch3nh2 | Methylamine | 40 / 150 | -95.068997659482 | -95.070748755162 | 1.751096 | 6.051e-10 | 4.818e-10 | 5 | 29.23 | Passed |
| W4-11/ch3f | Fluoromethane | 32 / 126 | -138.578628682950 | -138.583671400810 | 5.042718 | 1.191e-09 | 2.635e-09 | 5 | 27.56 | Passed |
| W4-11/h2co | Formaldehyde | 28 / 114 | -113.568403507374 | -113.571068435209 | 2.664928 | 1.182e-09 | 7.814e-10 | 5 | 26.67 | Passed |
| W4-11/hcn | Hydrogen cyanide | 24 / 102 | -92.682887704173 | -92.684481785537 | 1.594081 | 1.419e-10 | 8.057e-10 | 5 | 26.10 | Passed |
| DC13/C6H6 | Benzene | 84 / 342 | -230.437666980512 | -230.440773266123 | 3.106286 | 2.004e-09 | 2.607e-10 | 5 | 42.46 | Passed |
| W4-11/f2 | Fluorine | 20 / 90 | -196.974698848325 | -196.984466743173 | 9.767895 | 5.025e-10 | 6.573e-10 | 5 | 22.23 | Passed |
| W4-11/co2 | Carbon dioxide | 30 / 135 | -187.106562196966 | -187.111167481313 | 4.605284 | 2.863e-09 | 1.532e-09 | 5 | 27.29 | Passed |
| BHROT27/acetamide_RC | Acetamide | 60 / 240 | -207.530233988068 | -207.534675171401 | 4.441183 | 7.560e-10 | 6.833e-10 | 5 | 37.24 | Passed |
| PA26/gly | Neutral glycine | 70 / 285 | -282.165658275817 | -282.171987413623 | 6.329138 | 9.451e-10 | 1.737e-09 | 5 | 40.56 | Passed |
| S22/02 | Water dimer | 36 / 138 | -151.633028632653 | -151.637358537512 | 4.329905 | 1.017e-09 | 3.960e-10 | 5 | 26.40 | Passed |
| S22/08 | Methane dimer | 52 / 186 | — | — | — | — | — | — | 24.12 | Failed |
| S22/02a | Water monomer | 18 / 69 | -75.813233627498 | -75.815390715359 | 2.157088 | 5.432e-10 | 2.258e-10 | 5 | 23.14 | Passed |
| S22/08a | Methane monomer | 26 / 93 | -40.159942675918 | -40.160749569145 | 0.806893 | 6.626e-10 | 1.208e-10 | 5 | 26.94 | Passed |

Failed record:

- `S22/08`: Nonfinite energy or gradient; the initial implicit gradient was not confirmed finite; contraction-output FD missing or failed; MACE-parameter FD missing or failed; 5-step Adam incomplete; energy decrease missing or negative; initial/final energies missing; incomplete Adam trajectory; SCF convergence record missing or failed.

## Supplementary final-gradient and direct comparisons

There were 3 supplementary final-parameter FD records, with 3 explicit passes; maximum absolute error `1.919e-06` Ha per unit direction.
The exact shell-direct/DF energy comparison covered 6 points, with maximum `|E_DF-E_direct|` of **1.170e-04 Ha**. This difference measures the Hamiltonian approximation and is not an AD error.

| ID | Final-parameter FD error | FD status | Number of direct points | Maximum DF/direct energy difference / Ha |
|---|---:|---|---:|---:|
| W4-11/h2 | 5.365e-08 | Passed | 2 | 9.698e-06 |
| W4-11/ch4 | 5.156e-07 | Passed | 2 | 7.399e-06 |
| W4-11/hf | 1.919e-06 | Passed | 2 | 1.170e-04 |

## Timing and unverified boundaries

Recorded CPU preparation times sum to `110.47` s (24/24 records contain timing fields); original scientific validation `661.42` s; reruns `0.00` s; supplementary checks `104.08` s.
The recorded work totals `875.97` s, including failed attempts. This is the sum of per-record computation times, including compilation, **not the Slurm job walltime**. Unrecorded loading, queueing, build, or diagnostic time is excluded.

- SCF convergence and orbital residuals verify stationarity only; RHF internal and spin-breaking Hessian stability were not evaluated in this round. F₂/N₂ cannot be described as verified stable ground states.
- The pilot records do not expose numerical adjoint/GMRES residuals; finite gradients and FD agreement do not replace direct residual records.
- No train/validation/test family split, shared-parameter training, ions/UHF, DFT, forces, or TDDFT validation was performed.
- Energy decreases are not interpreted as DZ/TZ accuracy, and the initial limited FD errors are not extrapolated to all final parameters.

## Provenance and reconstructability

- Structure-name inventory: `/Volumes/TF/GradSCF_GMTKN55/selections/hf_pilot_20261007/systems.csv`.
- Main results: `/Users/jiaoyuan/Documents/GitHub/GradSCF/artifacts/nnao-gmtkn55-pilot-20261007/job-93084/validation`.
- `metrics.csv` stores 24 results and normalized metrics; `verification.json` stores SHA-256 values for every file read, recorded remote import paths/source hashes, the original failure history, and replacement records.
- Input-file hashes were calculated when this report was generated; remote source/native-library hashes come from the run records. This script did not log in again or recheck remote files.
- Runs that previously imported code from other installation paths are not automatically merged into the main results.

Report reconstruction command:

```sh
python artifacts/nnao-gmtkn55-pilot-20261007/make_report.py --validation-dir '/Users/jiaoyuan/Documents/GitHub/GradSCF/artifacts/nnao-gmtkn55-pilot-20261007/job-93084/validation' --supplement-json '/Users/jiaoyuan/Documents/GitHub/GradSCF/artifacts/nnao-gmtkn55-pilot-20261007/job-93084/final_and_direct.json'
```
