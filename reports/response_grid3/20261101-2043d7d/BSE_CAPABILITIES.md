# Molecular DFT → GW → BSE capability and evidence

2026-11-01; source baseline `2043d7d`, worktree
`perf/bse-tddft-grid3-20261101`. The production BSE/GW implementation is unchanged.
The campaign requests SVWN, PBE, PBE0 and original wB97X-D, Cartesian basis/grid3,
and complete orbital spaces. Source support is not a claim that every requested
combination has passed an end-to-end numerical calculation.

## Actual small runs and current acceptance

The local raw JSON, manifests and process exits have been independently inspected.
Both processes exited 0 on Slurm allocation `114465`, node `n005`, CPU float64,
eight logical CPUs (`0,1,2,3,64,65,66,67`), BLAS/OpenMP 1, JAX/JAXlib 0.10.0.
Both used full SCF integrals, true independent `weigend` auxiliary RI for GW,
`nw=32`, grid3, Cartesian AO, complete QP/optical/screening spaces and TDA
Davidson (`bse_conv_tol=1e-9 Ha`). The source hashes listed in both manifests
match the unchanged GW/BSE/SCF files in this worktree.

| Actual input / scope | Results checked under the strengthened forward gate | Raw evidence |
|---|---|---|
| H2, 0.74 Å, STO-3G, RKS/PBE, 1 root; four fresh workflows | 4/4 pass; screening valid in every row; maximum BSE residual 0 Ha; finite energies/strengths; all QP levels covered | [JSON](raw/h2-rks-pbe-g0w0-bse.jsonl), [manifest](raw/h2-rks-pbe-g0w0-bse.manifest.json), [exit](raw/h2-rks-pbe-g0w0-bse.process.json) |
| H2O, manifest geometry, def2-SVP, RKS/PBE, 3 roots; synchronized preparation diagnostic | 1/1 pass; screening valid; maximum BSE residual 2.7418750045e-10 Ha; finite energies/strengths; 25/25 QP levels covered | [JSON](raw/bse-preparation.jsonl), [manifest](raw/bse-preparation.manifest.json), [exit](raw/bse-preparation.process.json) |

H2 workflow times were 9.72122/0.09710/0.09403/0.08984 s; the entire child
process, including imports and setup, took 13.85821 s. These timing scopes are
not interchangeable. The water diagnostic recorded GW 57.04132 s, BSE
2.29721 s, first reference preparation 0.20118 s, and two additional
`get_bse_inputs()` calls 0.002689/0.002252 s. Its whole child process took
70.71634 s. The synchronized diagnostic does not enter clean performance
ratios. These are bounded small-input observations, not scaling claims.

The ownership diagnostic also passed: two live GW exports produced two MO
transforms, while reusing the first explicit snapshot produced no additional
transform and preserved energies/oscillator strengths and live-source stale
checks ([log](raw/bse-preparation-test.log),
[exit](raw/bse-preparation-test.process.json)). This is a call-count and
ownership check, not measured memory savings.

There is no repeated BSE preparation inside one independent workflow. These
measurements do not justify a persistent BSE cache; no such production change
was made. Reusing an explicitly owned `BSEReference` can avoid preparation for
multiple consumers of one unchanged GW result, but cannot be used across the
independent workflows in the cold/warm timing protocol.

The probe's forward gate now requires all of:

- `screening_valid` is true;
- every requested root is converged and stable;
- all excitation energies and original solver residual norms are finite;
- every residual is at most the run's `bse_conv_tol`, in Hartree;
- oscillator strengths are finite for the completed optical workflow.

`response_valid` remains a separate prerequisite for isolated-root derivative
response; it is not an executed adjoint/finite-difference test.
`stability_certified` remains separate: full dense BSE can certify the complete
static A±B spectra, while Davidson uses numerical Ritz/residual stability checks.
TDA does not provide a global stability certificate. The five previous BSE rows listed above have been re-evaluated under this
strengthened gate; every row passes. Their producing probe hash was
`13730e16a4e4046531cae33c25d38e370e881231c0bb24c49bf4e7ce1f6fd82a`;
the gate-only revision is
`c8ad71f1cabe1c7a451436d74b4a10b85e497cf291d30f58c3bac146ca5d08ac`.
This raw-record audit is not a numerical rerun with the revised probe. The current diagnostic has not established external reference
accuracy, full-BSE/UKS coverage, or nuclear/parameter AD for this campaign.

## Source-verified chain support

All supported entries below require real orbitals, integer occupations, valid
screening gaps, converged actual GW output and QP coverage for every selected
BSE level. The listed starting functionals are accepted through the actual
complete KS Fock/potential, including hybrid/RSH contributions.

| KS / GW | Static BSE chain | Spectrum and orbital-frame contract | Current campaign evidence |
|---|---|---|---|
| RKS / G0W0 | Supported | Original orthonormal KS MO frame; actual starting W0 poles recorded separately from QP energies | PBE H2 and water small runs audited above; other requested XC source-only |
| RKS / evGW | Supported | Fixed KS frame; final updated screening spectrum; uncomputed QP coverage cannot be inferred | Source-only |
| RKS / qsGW | Supported by current facade | Returned rotated MO frame and final independent-particle screening spectrum; facade records full computed/converged masks | Source-only |
| RKS / scGW | Unsupported | Matsubara G/Σ has no validated continued QP poles; static Fock eigenvalues are not QP energies | Public export guard passed after actual tiny RKS/PBE DFT; no scGW solve or fabricated BSE result |
| UKS / G0W0 | Supported | Real collinear, spin-conserving channels; shared AO auxiliary metric; `singlet=None` | Source-only |
| UKS / evGW | Supported with an existing factor source | Same real spin-resolved frame; updated screening spectrum; independent `UGW.density_fit()` is not available for evGW | Source-only |
| UKS / qsGW or scGW | Unsupported | UGW facade implements G0W0/evGW/evGW0 only | Source-verified rejection |
| GKS / any requested GW | Unsupported | No spinor GW facade or complex/noncollinear molecular BSE adapter | Tiny actual GKS/PBE state rejected by GW, UGW and BSE source adapters; UKS is not substituted |

Optional evGW0 is a fixed-W0 control, not a substitute for requested evGW.
The old BSE README/FULL_BSE_PLAN limitations do not supersede the current qsGW
facade implementation.

Source pointers: [GW export and qsGW masks](../../../src/gradscf/gw/rgw.py),
[UGW method/RI boundaries](../../../src/gradscf/gw/ugw.py),
[qsGW returned-frame contract](../../../src/gradscf/gw/qsgw.py),
[SCGWResult without QP poles](../../../src/gradscf/gw/scgw.py),
[BSE reference and coverage validation](../../../src/gradscf/bse/reference.py),
[BSE public facade](../../../src/gradscf/bse/api.py).

## Solver, integral and storage boundaries

TDA and full BSE have dense reference and matrix-free Davidson routes. Davidson
avoids complete A/B matrices but retains bare MO factors. Static screening may
use direct Cholesky or matrix-free GMRES; direct mode already reuses its
Cholesky factor. Neither choice changes the physical static kernel or stops
screening-parameter derivatives. The eager facades are validation boundaries;
JAX transformations use the functional numerical drivers.

Independent auxiliary RI is distinct from factorizing full ERIs. With a full
SCF source, `--gw-backend scf` uses ERI spectral factors with tolerance `1e-12`;
it is not a conventional no-factor BSE implementation. Actual conventional
four-center GW has no static BSE export. UGW independent RI is restricted to
G0W0; its evGW path needs existing SCF factors or the full-ERI factor source.
Original wB97X-D does not gain an unsupported RI-JK SCF path through this probe.

For 974 auxiliary functions and 564 MOs, one float64 full MO factor array has
309,825,504 elements / 2,478,604,032 bytes; UKS has twice that MO-factor storage.
The default `max_aux=1024` passes, but `max_factor_elements=20,000,000` rejects
this size before transformation. Raising that cap requires a separate ownership
preflight; it is not a peak-memory guarantee. With 47 occupied and 517 virtual
orbitals, OO/OV/VV alone occupy 17,212,528 / 189,337,808 / 2,082,715,888 bytes.
Removing duplicate VO would save only 7.6389% and requires preserving the current
symmetric-input AD mapping. No such representation change was made.

Relevant checks exist in `tests/bse/test_api.py`, `test_gw_reference.py`,
`test_unrestricted.py`, `test_full.py`, `test_matrix_free.py`,
`test_screening_memory.py`, `tests/xc/test_gw_starting_points.py` and
`tests/test_gw_bse_compile_reuse.py`. Listing them is not a claim that this
campaign has rerun them. The additional preparation/ownership fixture is
`benchmarks/probes/response_grid3/test_bse_preparation_baseline.py`.

The public rejection checks now have runtime evidence: two tests passed,
zero failures/errors/skips, on `n005` ([JUnit](raw/bse-runtime-guards.xml),
[process exit](raw/bse-runtime-guards.process.json)). The test first solves tiny
RKS/GKS PBE/grid3 DFT references, then exercises the public scGW export and GKS
GW/UGW/BSE source guards. The restricted GW attempt on GKS currently raises a
missing-restricted-SCF-payload `RuntimeError`; UGW reports unsupported spinor
response. This proves those adapter refusals, not scGW convergence or analytic
continuation, and does not fabricate a QP spectrum.
