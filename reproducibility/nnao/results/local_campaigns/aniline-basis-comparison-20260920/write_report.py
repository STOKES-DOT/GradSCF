"""Report completed small-basis validation separately from pending large jobs."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
p=json.loads((root/'preflight.json').read_text())
results=[]
for folder in ('remote-smoke','remote-polarized'):
 result_path=root/folder/'comparison.json'
 if result_path.exists():results.extend(json.loads(result_path.read_text())['results'])
energies={r['basis']:r['energy_hartree'] for r in results}
lines=['# Aniline basis comparison — preflight and validated small-basis results','',
'Prepared on 2026-09-20 against the current GradSCF module layout. This report is not the completed eleven-basis or NNAO training result.','',
'## Geometry and Hamiltonian','',
'Aniline C6H7N, 14 atoms, 50 electrons, neutral singlet, 25 occupied spatial orbitals. All calculations use the same saved geometry, RHF, spherical AOs and float64. The fixed geometry was generated using RDKit ETKDGv3 (seed 20260916) followed by MMFF94 minimization; it is not a quantum-optimized geometry. Original source SHA256: `'+p['geometry_source_sha256']+'`.','',
'## Current module layout','',
'- NNAO: `gradscf.model.nnao` (`src/gradscf/model/nnao/`).',
'- Integrals: `gradscf.integrals`.',
'- Vendored MACE runtime root: `src/gradscf/model/nnao` for importing `mace_jax`.',
'- The fresh remote snapshot is `/tmp/gradscf-aniline-20260920`; it does not reuse the former top-level `src/nnao` package. Native source hashes match the previously validated compiled Linux library.','',
'## Sizes and completed energies','',
'| Basis | Contracted AOs | Primitive AOs | Fock matrix | Contracted ERI / GiB | Primitive ERI / GiB | RHF energy / Ha |',
'|---|---:|---:|---:|---:|---:|---:|']
for r in p['rows']:
 e=f"{energies[r['basis']]:.12f}" if r['basis'] in energies else 'Not run'
 lines.append(f"| {r['basis']} | {r['nao']} | {r['primitive_nao']} | {r['nao']} x {r['nao']} | {r['contracted_eri_gib']:.3f} | {r['primitive_eri_gib']:.3f} | {e} |")
lines += ['',
'Tensor storage is calculated as 8*N^4 bytes and is not the process peak. Standard RHF uses contracted ERIs. The existing coefficient-training path caches primitive ERIs and differentiates their JAX contraction, requiring much more memory. The current spectral factorization also starts from a full ERI matrix and does not remove the allocation.','',
'## Verification','',
'Local focused regression: 36 passed, 2 optional MACE dependency skips; the self-contained molecular tests were then rerun (4 passed) after removing their artifact dependency. The final molecular/dependency checks had 6 passed, 1 optional skip. The new tests cover C/H/N model assembly, a nitrogen-containing molecule\'s energy gradient, and pre-allocation guards. Remote outcomes are in remote-tests.log. Remote focused regression: 33 passed in 67.12 seconds, including the current C/H/N MACE assembly. SCF results and independent PySCF checks are in remote-smoke/, remote-polarized/ and their reference logs.','',
'## Full run awaiting a resource decision','',
'`run_full.sh` is prepared but has not been launched. It selects 8 CPU cores, a 700 GiB process virtual-memory limit, and requires more than 750 GiB available memory before starting. This limit is a ceiling, not a measured peak-memory estimate. The current remote host is c20; its availability must be rechecked at launch. Nine standard bases and the two NNAO families run serially; NNAO coefficients are optimized afresh on aniline with seed 0, maxiter=600, gtol=5e-7 and ftol=1e-14. The maximum iteration count is not a convergence guarantee. Wall time and peak memory have not been measured.','',
'The high-memory job requires user approval under the user-provided Global AGENTS.md instruction to ask before long remote jobs that consume noticeable resources. An alternative is to implement and validate a blockwise/low-memory integral path first. No methane-trained coefficients or energies are presented as aniline results.','']
(root/'README.md').write_text('\n'.join(lines))
