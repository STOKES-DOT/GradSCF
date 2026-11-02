# NNAO core basis benchmarks

## Scope

Three comparisons use one frozen MACE checkpoint (update 3351), native
DF-HF/def2-universal-jkfit, spherical AOs and float64. They are new HF
benchmarks, not a reproduction of the Grimme DFT numbers.

| Measurement | Full cases | Systems |
|---|---:|---|
| Atomic radial electron density | 36 | H, H-, C, C-; eight bases plus aug-cc-pV5Z references |
| Counterpoise BSSE | 176 | All 22 S22 dimers, eight bases, five energies per case |
| Static polarizability | 72 | Nine molecular identities, eight bases, three analytic field responses |

Traditional bases are STO-6G, STO-6G+P, 6-31G(d,p), def2-SVP, cc-pVDZ,
cc-pVTZ and ma-def2-QZVPP. STO-6G+P uses original def2-SVP polarization.
See [RUN_SPEC.md](RUN_SPEC.md) for conventions, geometry sources, tolerances
and exact commands. Comparison implementation: `tests/comparisons/nnao_core/`.

## Verification

Sixteen focused local checks passed, including independent PySCF atomic UHF,
H2 CPHF, and all five He2 counterpoise energies using a common dimer auxiliary
basis. No finite-difference checks were performed.

Remote smoke job 101883 completed 12/12 cases on a V100. The numerical loop
took 67.36 seconds, excluding startup, imports and checkpoint loading.
These are pilot results, not full S22 or nine-molecule averages:

| Pilot measurement | STO-6G+P | def2-SVP | NNAO/MACE |
|---|---:|---:|---:|
| Water dimer BSSE / kcal mol^-1 | 4.7403 | 1.8806 | 0.2013 |
| Ethylene isotropic polarizability / Bohr^3 | 13.1027 | 21.4858 | 15.3792 |

Pilot plots and raw results are in `smoke/`. H/H- are charge/spin/isolated-atom
extrapolations from molecular training. The model has no electric-field input;
polarizability means SCF response in the fixed predicted basis.

## Full submission

The user approved the full campaign on 2026-10-20. Slurm job **101888** requests
one A100 80G, 32 CPU, 512 GiB and a three-hour limit. It completed on n001 at
2026-10-20 14:45:14 +08:00, after 2 h 42 min 30 s, with Slurm exit code 0:0.
The benchmark status is `complete_with_failures`: all 284 cases were attempted,
272 passed and 12 failed. A successful Slurm exit does not mean every case passed.
Submission evidence is in `submission.json`; final case results are in
[`full/results.json`](full/results.json).

| Measurement | Passed | Failed |
|---|---:|---:|
| Atomic radial electron density | 36 | 0 |
| S22 counterpoise | 166 | 10 |
| Static polarizability | 70 | 2 |

All 35 NNAO/MACE cases passed (4 atomic states, 22 dimers, 9 polarizabilities).
The six conventional bases other than ma-def2-QZVPP each passed all 35 cases;
the four aug-cc-pV5Z atomic references also passed. All 12 failures are GPU
out-of-memory errors for ma-def2-QZVPP: ten S22 cases and the naphthalene and
anthracene polarizabilities. They are retained explicitly, without replacement
or claims of complete QZ reference coverage.

Final figures: [atomic radial density](full/atomic-radial-density.png),
[S22 counterpoise](full/s22-counterpoise.png), and
[static polarizability](full/static-polarizability.png).
The S22 aggregate plot uses the same 12 successful dimers for all eight bases;
the full 22-dimer results for the other bases remain in `full/s22-cases.csv`.
Missing ma-def2-QZVPP polarizabilities appear as gaps.

Remote working directory:
`/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014`.
Runtime files are written under `artifacts/nnao-core-benchmarks-20261020/full/`;
the Slurm log is `full-101888.log`. Each case is saved independently, with
failures retained explicitly. Resume requires identical configuration and
source hashes. No production sources or training checkpoints were changed.

Generate figures after downloading the case outputs:

```bash
python tests/comparisons/nnao_core/plot.py \
  artifacts/nnao-core-benchmarks-20261020/full \
  --dataset artifacts/nnao-core-benchmarks-20261020/dataset.json
```
