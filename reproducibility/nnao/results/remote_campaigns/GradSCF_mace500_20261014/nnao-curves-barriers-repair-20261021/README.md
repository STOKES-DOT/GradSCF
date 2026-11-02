# Native HF repair continuation — 2026-10-21

Job 102347 reached its three-hour time limit: 216/242 cases evaluated,
209 accepted, 7 rejected by real internal stability, 26 not processed.
All 21 original noncovalent failures were recovered. This continuation
keeps those 209 accepted records and requests exactly the other 33 cases.

## Root cause and narrow production fix

In unrestricted DF exchange, orbital hints were used even when they did
not represent the supplied density. A density-only restart consequently
combined Coulomb from the new density with exchange from core orbitals.
The independent-density and mixed factor/density AD responses were also
incorrect. UKS now reuses RKS's existing consistency-checked DF helper
and its physical-density custom JVP. No new orbital-relaxation fallback
is introduced.

HO2/STO-6G reproduced the error: the saddle had energy
-149.594788410214 Eh and curvature -0.039864870969; a directly minimized
determinant had energy -149.615257608827 Eh and orbital residual
3.96e-14, yet DIIS returned to the saddle before this fix.
The patched regression requires a stable, lower-energy result.

## Provenance

The original benchmark and retry reports are retained. Remote UKS is
patched surgically by `patch_frozen_uks.py`, after verifying the original
scientific source and checkpoint hashes. `source-before/uks.py`,
`uks.patch`, and `source-patch.json` record the change. A separate
`snapshot.json` identifies the patched source; checkpoint update 3351,
exported basis functions, exponents, geometries, and DF auxiliary basis
remain identical. Accepted records retain their original source lineage.

## Validation and execution

Local CPU/float64 focused gate: **44 passed**, 1 PySCF OpenMP warning,
35.83 s. It covers native seeds/stability, HO2 escape, DF hint values,
density JVP and mixed higher derivatives, retry merging, protocols,
counterpoise, and field-response comparisons. No finite differences.

`continue.slurm` requests one A100 80G, 32 CPU, 512 GiB, at most 2 h.
It runs a node-local regression gate, computes 33 cases, combines all
242 retries, merges them into the separate 852-case report, generates
figures, and independently checks 31 same-basis energies with PySCF
(including internal stability for 13 selected cases). PySCF is only an
independent comparison, never a production solver fallback.

Real internal stability does not certify the global HF minimum or
external spin/complex stability. Failed checks remain explicit. No full
four-index ERI is stored, and no training task or checkpoint is changed.

## Completed remote run and additional local checks

Job **102690** ran on n001 / A100 80GB PCIe, from 2026-10-21
10:35:59 to 11:10:16 (UTC+08:00): 34 min 17 s for the complete script.
The node-local gate passed 17/17. The 33 requested cases passed 33/33
in 971.03 s, recovering all seven previous stability failures.
All 209 previously accepted records were verified unchanged.

Combined retries: **242/242 passed**. The separate merged benchmark
contains **852/852 passed**: S66x8 288/288, HB300SPXx10 360/360,
and BH9 204/204 species, producing 54/54 reaction/basis profiles.
The independent comparison passed **31/31** with maximum absolute
energy difference **2.09184e-11 Eh**; all 13 requested independent
internal stability checks passed. No independent run retained full ERI.
The independent checks took 1003.80 s in total.

Reports and regenerated PNG/PDF plots are downloaded into `merged/`
and `combined-retry/`. The three overview plots were visually inspected.
Pass counts concern numerical convergence/physical residuals and the
specified stability checks, not agreement with correlated references.
For these nine BH9 reactions, NNAO MAEs against DF-HF/cc-pVTZ are
8.40 / 6.01 / 12.74 kcal/mol for forward / reverse / reaction energies.
Slurm accounting is disabled, so completion is established by all
final report states and the log reaching the script's final timestamp.

Patched scientific-source SHA256:
`8c77b02a219296e4085a29a977529f2ef2e8817584d39c880675747b98be05f0`.

Additional isolated runs passed: `test_scf_uhf.py` 7/7 and
`test_uhf_stability.py` 3/3. `test_scf_uks.py -k 'not b88'` had 8 passed,
2 skipped, 1 deselected. The B88 empty-beta test could not run because
local `jax_xc` is absent. A broader combined process also terminated
with a JAX lowering/garbage-collection segmentation fault; the isolated
UHF runs passed afterward. The complete SCF suite is not certified.
