# Changelog

## Unreleased — GradSCF

- Remove the `gradscf.traditional_xc` compatibility namespace (use
  `gradscf.dft` / `gradscf.dft.xc` directly), and move the XC backends from
  `gradscf.xc_backend` to `gradscf.dft.libxc_jax`.

- Move the neural-network model code into the new `gradscf.model` subpackage:
  `gradscf.neural_xc` -> `gradscf.model.neural_xc`, `gradscf.neural_d` ->
  `gradscf.model.neural_d`, `gradscf.training` -> `gradscf.model.training`,
  and the top-level `nnao` package -> `gradscf.model.nnao`.  The vendored
  mace-jax project remains a separate top-level package (sources unchanged).
  Top-level `gradscf` symbols (e.g. `Functional`, `make_neural_xc_functional`,
  `MolecularTrainingConfig`) are unchanged.  No compatibility aliases are
  kept for the old import paths.

- Rename the distribution and Python namespace from `td-graddft` / `td_graddft`
  to `gradscf`; rename `td_graddft_tools` to `gradscf_tools`.
- Update active imports, dynamic module paths, package-data declarations,
  examples, command-line tools, and contributor documentation.
- Retain scientific method names, upstream GradDFT names, historical
  reproducibility artifacts, and the existing NPZ target-bundle format marker.
- Retire the old import namespaces without a compatibility alias.

### Molecular GW and BSE

- Add real closed-shell static singlet/triplet TDA-BSE with factorized Davidson
  actions and a bounded dense oracle. Reuse shared eigensolvers and screened
  linear solves; include isolated-root energy and optical-property response.
- Add bounded stable full BSE (`tda=False, solver="dense"`) with coupling
  blocks, metric-normalized X/Y and first-order amplitude response through a
  shared Cholesky-Hermitian RPA solver. Report stability margins and reject
  invalid derivatives; scalable full-BSE Davidson remains deferred.
- Record QP-computation coverage and the actual screening spectrum in CD GW
  results. Add checked G0W0/evGW result snapshots for fixed-frame BSE inputs,
  without claiming evGW outer fixed-point differentiation.
- Add native water TDA/full-BSE examples, pinned QuAcK Fortran A/B kernel
  fixtures, TDHF limit comparisons and first-order response regressions.


### SCF and native integrals

- Add RHF/UHF, ROHF/ROKS, and GHF/GKS workflows and opt-in UHF/UKS
  internal-stability analysis with bounded lower-energy restarts.
- Consolidate convergence, DIIS, and energy/Fock assembly; reuse converged
  RKS results when preparing response references.
- Unify implicit/unrolled SCF differentiation and force-training helpers;
  retain only the current XC binding protocols.
- Consolidate integral code, full basis assets, and vendored native sources
  under `gradscf.integrals`; remove production PySCF/GPU4PySCF calls.
- Add native coordinate JVP/VJP and mixed force/parameter differentiation.
  Native basis derivatives and pure coordinate Hessians remain unsupported.

### Periodic calculations

- Add neutral 3D GTH Gaussian HF/DFT at Gamma and uniform k meshes, with
  FFT density fitting and q=0 TDA/TDDFT. Complex k-point full TDDFT uses
  a bounded dense solver; metals, finite-q response, and low-dimensional
  electrostatics are not supported.
- Reject cached periodic response results after cell or FFT mesh changes
  until the SCF reference is recomputed.
- Add fixed-density LDA/GGA band queries and restricted Gamma velocity-gauge
  oscillator strengths including the nonlocal GTH commutator.
- Add reproducible silicon/diamond band and silicon Gamma TDDFT spectrum
  comparisons with PySCF. Gamma strengths are not macroscopic absorption
  coefficients; basis and k-mesh convergence are not claimed.

## 1.0.0 - 2026-08-24

### Features

- Provide differentiable restricted and unrestricted SCF with explicit and
  implicit gradient modes.
- Provide restricted and unrestricted TDA/Casida response solvers, Davidson
  eigensolvers, and differentiable Neural XC response paths.
- Support semilocal channels plus fixed-cache `nograd` local-HF and optional
  PT2/CIS(D)-style Neural XC channels. Ground-state HFX/PT2-SCF recomputation is
  not part of v1.0.0.
- Support CPU/libcint, JAX, density-fitting, and GPU4PySCF integral backends.
- Provide the manuscript training and evaluation entry points for H2+, H2, N2,
  QM9 ground-state, and QM9GWBSE S1/TDA tasks.

### Reproducibility

- Add selected manuscript checkpoints, final inference CSV/JSON files, compact
  reference tables, final figures, and a SHA-256 manifest under
  `reproducibility/v1.0.0/`.
- Validate the published QM9 ground and QM9GWBSE S1 checkpoints with the v1.0.0
  inference API.
- Keep only manuscript-facing examples and tools, and document conventional
  DFT/TDDFT plus Neural XC training, checkpoint, and inference workflows.
- Normalize public QM9 naming and remove machine-specific paths from released
  reproducibility metadata.

### Fixes

- Align the standalone JAX molecular grid with PySCF 2.13 for levels 0-9:
  full Lebedev tables through 1454 points, Treutler-Ahlrichs radial constants,
  NWChem pruning, Becke partitioning, and the PySCF Angstrom-to-Bohr constant.
  Keep the grid JIT- and geometry-gradient-safe without caching traced arrays.
- Solve implicit-SCF adjoint systems with JAX GMRES and remove the previous
  regularized fixed-point residual bias.
- Align the matrix-free full-TDDFT Davidson solver with PySCF-style dual
  subspaces, residual-only convergence, and differentiable symplectic Rayleigh
  reconstruction.
- Extend restricted and unrestricted TDA/TDDFT validation to end-to-end JAX
  SCF references, including excitation energies and degenerate-cluster
  oscillator strengths.
- Use canonical `excitation_gap_*` metric names in closed-shell checkpoint
  evaluation summaries.
- Keep restricted two-AO references out of unrestricted spin-axis dispatch and
  keep occupation conversion JIT-safe.
- Remove obsolete training loss/RSH modules and legacy research drivers from
  the public release tree.
