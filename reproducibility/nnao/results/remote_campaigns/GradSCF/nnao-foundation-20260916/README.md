# NNAO basis assembly validation — 2026-09-16

Implemented MACE invariant features -> per-atom s/p contraction coordinates ->
GradSCF basis parameters, plus native-primitive/JAX contraction helpers.
Includes 34 main-group SZP3 initial templates through Xe and molecular element/
grid support through Z=54. The basis is an unoptimized starting point, not an
SCF-trained model. Upstream MACE source checksums are unchanged.

## Verified checks

- Fresh local targeted gate: **59 passed, 5 skipped**. Skips: real MACE stack
  absent locally and four PBE cases because local jax_xc is absent.
- Fresh remote gate: **31 passed in 125.89 s**, CPU float64, affinity 0–3.
  Includes real MACE JIT, rotations/permutations, all-element zero-head
  initialization, gradients to encoder parameters, parameter finite differences,
  Cartesian/spherical integral reference tests for H–F/Cl/Br/I, and 8 converged
  HF/PBE calculations compared with independent PySCF references.
- HF/3-21G tolerance: 1e-8 Ha. PBE/3-21G tolerance: 2e-7 Ha, grid level 1.
  Hydride bond lengths F/Cl/Br/I: 0.92/1.27/1.41/1.61 Angstrom.
- Native library rebuilt from this source snapshot on Linux. 683 source and
  resource hashes verified before the run and matched to the local files after.
- Wheel build succeeded. Current adapters, basis data and native source resources
  are included; upstream MACE tests/examples and local binaries are excluded.
- No full-suite, GPU, SCF-training, ECP or relativistic accuracy claim.

## Environment

Base Python: /home/yjiao/opt/miniconda3/envs/jax_scf/bin/python.
JAX/JAXlib 0.10.0; Flax 0.12.7. Remaining versions are in metadata.json.
The base environment lacks torch and e3nn. This isolated test reused PyTorch
2.9.1 from grad_dft and installed only e3nn 0.4.4 plus opt-einsum-fx 0.1.4
in /tmp/gradscf-nnao-20260916/deps. The base environments were not modified.
The test selected the pinned vendored MACE source ahead of the installed package.

## Provenance and limitations

Final isolated source: /tmp/gradscf-nnao-verified-20260916.
An earlier overlay of the old remote GradSCF failed because it inherited obsolete
SCF APIs and missing grid files. It was superseded by a fresh, hash-verified
snapshot and native rebuild. Its failures are not included in the passing gate.

See src/nnao/NNAO.md for API, units, initial parameter provenance and limitations.
Full primitive ERI caching is a small-system validation path. High-level SCF/grid
assembly remains Cartesian-only; spherical support here is for integral plans.
