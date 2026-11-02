# Dimer CP × MACE-output controls — execution record

## Completed handoff, 2026-10-27

All **204/204 new cases passed**, with no numerical failures. The merged data
contain 864 five-energy records and 1728 calculated interaction values plus
216 reference values. Both figure sets (24 individual figures and two 12-page
PDFs) were generated, numerically verified and installed in NNAO_paper at
10:48 +08:00. See `INSTALL_REPORT.json` and `visual-qa.json` for the installed
path, checksums, per-panel range checks and scoped visual inspection.

Delivery watcher session 99323 finished successfully after installation.
Publication text, existing figure scripts and previous archives were not
edited. The chronology below retains the original submission observations.

User accepted the preceding comparison on 2026-10-26. The definition is fixed
at scale-1.00 coefficients versus archived geometry-adaptive coefficients;
network weights, primitive exponents, checkpoint update 3351, twelve dimers
and 108 source geometries are unchanged. See `RUN_SPEC.md`.

New work: 204 five-energy cases, 1020 independent SCF energy problems before
initial-guess/rescue attempts. Reuse: 660 cases. Final comparison: 864 records,
1728 calculated interaction values plus 216 displayed reference values.
Both CP and no-CP panels include fixed/adaptive MACE, STO-6G, STO-6G+P,
6-31G(d,p), def2-SVP, cc-pVDZ, cc-pVTZ and upstream reference curves.

## Current pipeline

- **109067**: CPU short gate, four actual missing cases. Completed 4/4 with
  all five SCF/stability checks passed for each. Those cases are retained.
- **109068**: continuation on n004 / A800, 32 CPUs, 128 GiB, three-hour limit;
  observed running from 2026-10-26 18:47:59 +08:00. Last directly observed
  state at 18:58 was 22/204 passed, no failed cases.
- **109220**: user-approved serial continuation, `afterany:109068`, one A800,
  32 CPUs, 128 GiB, at most three additional hours. It skips all SCF if the
  parent has already completed 204/204; otherwise it resumes only unsaved cases.
  Arranged on 2026-10-26 with 132 passed records retained and zero failures.
- **109991**: final 21-case continuation, explicitly approved on 2026-10-27;
  one A800 / 32 CPUs / 128 GiB, two-hour limit. Observed RUNNING on n004 from
  09:57 +08:00, retaining 183 passed cases with zero numerical failures.
  The prior 109068 and 109220 each reached their three-hour time limit.
  Remaining groups: seven cyclopentane-dimer and fourteen water-dimer cases.
- **109078**: lightweight CPU merge/plot/verify job, updated to
  `afterok:109991`.
- `watch_delivery.py`: finite eight-hour local watcher, running via exec session 99323;
  state is recorded in `delivery-progress.json`. It waits for verified
  figures, downloads/checks the archive and installs only a new data package:
  `/Users/jiaoyuan/Documents/GitHub/NNAO_paper/data/computational_examples/2026-10-26/cases/15_dimer_cp_output_controls/`.
  It adds narrow data-index entries and does not edit publication text, old
  figures/scripts, existing archives or training jobs. It stops for review
  on calculation/postprocessing failure, rather than publishing partial data.

`INSTALL_REPORT.json` is produced only after actual successful installation.
The watcher records that model visual review is still unclaimed; inspect the
downloaded images before claiming that review. Numeric and legend/inset
layout checks are automatic. No new final figures existed at launch time.

## Paths and validation

Remote root:
`/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014/`.
Remote artifact directory matches this relative directory. Raw status is in
`run/progress.json`, `run/results.json`, `run/gate.json`; per-process resource
records are in `run/resource-<job>.json`.

Source SHA256 remains
`8c77b02a219296e4085a29a977529f2ef2e8817584d39c880675747b98be05f0`.
Only new comparison drivers and artifact files were uploaded; no production
source, weight or existing result was modified. Host protocol checks passed
13/13. No finite differences were performed.

Initial launch 109058 exited before SCF because `/usr/bin/time` is absent on
the node. Its blocked dependent 109059 was replaced, and Python standard
`resource`/`time` monitoring was verified and used for 109067/109068.
The failed launch did not contribute numerical records. All logs and old
submission metadata remain available under the remote artifact directory.
