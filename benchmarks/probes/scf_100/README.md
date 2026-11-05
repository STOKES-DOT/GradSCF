# SCF-100 execution benchmark

The fixed physical task is the existing beta-carotene PubChem conformer: C40H56,
96 atoms, 296 electrons, Cartesian def2-SVP and KS grid level 3. This is one
approximately 100-atom molecule, not another molecule sweep.

`make_plan.py` uses only the standard library. It freezes 30 method/representation
rows per engine: RHF/UHF/GHF and RKS/UKS/GKS with SVWN, PBE, PBE0 and CAM-B3LYP,
each with explicit weigend RI or conventional direct J/K. Pure XC needs RI-J;
HF/global hybrids request RI-JK; RSH requests RI-J plus exact direct exchange via
`density_fit(auxbasis='weigend', only_dfj=True)`. No four-center spectral surrogate,
packed fallback, spin-method substitution or functional substitution is allowed.

Use the registered `reports/scf_100/20261105-40a569bf/plan-v2/` inputs for the
baseline and candidate. `plan/` is an unexecuted preparation draft preceding the
explicit `only_dfj` contract; its superseding revision is documented in the report.

```bash
python benchmarks/probes/scf_100/make_plan.py /new/scf-100-plan
```

A separate small smoke input is generated without changing the large task:

```bash
python benchmarks/probes/scf_100/make_plan.py /new/scf-100-smoke \
  --system methane --basis sto-3g --engines gradscf
```

Run scientific commands only in an allocated Slurm step, from the cluster's
required logical homefile directory. The following command is the worker smoke;
use an empty output path and the existing process-group guard. Environment imports,
basis-dimension preflight and final certificates are excluded from the facade timer
and included in the outer guard. The guard samples actual process-group RSS at
0.2 s; software `max_memory` is a planning budget, not a hard RSS ceiling.

```bash
PYTHONPATH=src:. python benchmarks/probes/response_grid3/guard.py \
  --seconds 24120 --rss-bytes 214748364800 /new/methane-smoke.process.json -- \
  python benchmarks/probes/scf_100/worker.py \
  /new/scf-100-smoke/configs/methane-sto-3g-rks-pbe-df-gradscf.json \
  /new/methane-smoke
```

The equivalent first large baseline uses:

```bash
PYTHONPATH=src:. python benchmarks/probes/response_grid3/guard.py \
  --seconds 24120 --rss-bytes 214748364800 /new/beta-carotene-pbe-ri.process.json -- \
  python benchmarks/probes/scf_100/worker.py \
  reports/scf_100/20261105-40a569bf/plan-v2/configs/beta-carotene-def2-svp-rks-pbe-df-gradscf.json \
  /new/beta-carotene-pbe-ri
```

All registered calls use at most 64 allocated logical CPUs, 180000 decimal MB
software planning budget, a 200 GiB RSS guard, and 6000 s per facade call. Ordinary
rows have one cold call; RKS/PBE and UKS/PBE0 have three additional new-object warm
calls for each representation. Their whole-worker limit is 24120 s; other rows use
6120 s. These are process limits, not permission to overlap heavy workers.

Both engines receive `atom`, but their initial density algorithms differ:
GradSCF sums native unspherical isolated-UHF atomic densities, while PySCF uses its
atomic-HF implementation. Fixed-input contraction diagnostics must be reported
separately if this initial-state difference is to be eliminated.

The PySCF worker creates a process-private `PYSCF_CONFIG_FILE` before importing
PySCF, setting `df_df_DF_lindep=1e-12`. It checks the actual bound DF default and
records the configuration hash. This is a reference configuration, not a patched
integral engine. PySCF's existing low-threshold accuracy caveat remains subject to
numerical acceptance. The no-RI reference can choose in-core exact ERIs when small
enough; such a run is recorded `not_comparable` to the requested direct workflow,
rather than being silently called direct. RI-J/direct-K with retained reference
ERIs is likewise recorded as `ri_j_packed_k` and excluded from a strict direct-K
comparison. The measured `ngrid` is the actual observed size, even for HF
references that prepare grids for later neural-XC work; unknown sizes stay null.
Metadata inspection does not retain RI/grid payloads between warm calls.

Missing public capabilities and unforwarded tolerances produce `unsupported`.
There is no hardcoded GKS/RSH skip: newly implemented public capabilities are
exercised by the same worker. The outer guard's successful process status does not
imply convergence. Require all requested repeats, finite outputs and the registered
E/D/MO-gradient gates, then apply the paired energy gate of 1e-7 Ha.

For invasive stage profiling, use `benchmarks/probes/scf_solver/profile_stages.py`
in a separate process with the same input. Its instrumentation timing is not a
clean speed measurement; it predates this protocol and must be checked for the
specific new capability before use. The clean worker itself has no timing hooks in
production numerical functions.

Metadata regression (no numerical libraries or scientific calculations):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  benchmarks/probes/scf_100/test_protocol.py
```
