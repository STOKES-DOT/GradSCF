# Dimer CP × MACE-output comparison

Approved definition: user instruction on 2026-10-26, following the proposed
four-group comparison. Frozen output means the Gaussian contraction
coefficients predicted at distance scale **1.00**, held for the entire scan.
Adaptive output means the archived MACE prediction at each geometry. Network
weights and shared primitive exponents remain unchanged in both policies.

## Scientific identity

- Frozen checkpoint update **3351**, original checkpoint SHA256
  `a7144d211a7b3d9fc4cb6899bf930072992e073bbc79d8cae282611682bc7c1f`.
- Existing repaired native source
  `8c77b02a219296e4085a29a977529f2ef2e8817584d39c880675747b98be05f0`.
  Production code, network weights, existing results and training are not edited.
- Existing 12 representative dimers and 108 source geometries: six S66×8
  dimers, six HB300SPX×10 dimers; no interpolation or geometry changes.
- DF-RHF, spherical AOs, float64, `def2-universal-jkfit`, identical Hamiltonian
  and source geometry/state labels. Five calculations share one fixed dimer
  auxiliary basis/metric; own-fragment DF factors are AO slices.
- SCF: 200-cycle attempts, energy 1e-12 Eh, density RMS 1e-10, MO gradient
  1e-9; raw-Fock orbital residual/idempotency/electron-count gates 1e-7.
  Existing native MINAO/shifted rescue and real internal-stability checks are
  retained for all five newly evaluated energies. No PySCF solver fallback,
  finite differences, full four-index ERI or native derivative compilation.

## Four comparisons and controls

At each geometry select either `c(R0)` or `c(R)`. That selection is then held
fixed among the five dimer/own/ghost SCF calculations. Electrons/orbitals
relax self-consistently in each calculation.

| Plot set | Fixed MACE output | Adaptive MACE output |
|---|---|---|
| No CP | E_AB - E_A(own) - E_B(own), using c(R0) | Same definition, using c(R) |
| CP | E_AB - E_A(ghost) - E_B(ghost), using c(R0) | Same definition, using c(R) |

The independent-monomer `model_interaction_kcal_mol` is not used in this
matched CP comparison. Original data remain archived separately.

Each plot includes STO-6G, **STO-6G+P**, 6-31G(d,p), def2-SVP, cc-pVDZ,
cc-pVTZ, both MACE curves and the original reference curve. Reference energies
are correlated-source benchmark values, so differences also include method
errors. The T-shaped benzene dimer's `TS` label is not a transition state.

STO-6G+P retains every STO-6G shell and adds the most diffuse original
uncontracted def2-SVP p shell on H, d shell on other elements. Explicit
reference exponents: H/C 0.8; N 1.0; O 1.2; F 1.4; P 0.45; S 0.55;
Cl 0.65; Br 0.389. Br's tighter occupied d radial functions are not added.
All source exponents/coefficients are preserved without fitting.

## Work and execution

There are **204 new five-energy cases**: 96 fixed-output MACE points plus
108 STO-6G+P points, **1020 SCF energy problems**. Initial-guess/shifted rescue
attempts within each energy problem are recorded separately and can increase
the actual iteration/kernel-call count. The 12 MACE reference anchors
reuse the identical adaptive records. In total 660 cases are reused and
864 five-energy records produce 1728 plotted interaction values (two panels
per system, eight calculated curves per panel). Upstream reference values
are added separately.

Host-only regressions passed 13/13, covering coefficient freezing, geometry
preservation, shared exponent pools, missing references/polarization assets,
CP energy selection, ghost nesting and packed-DF slicing.

A short, 32-CPU / 64-GiB / 30-minute CPU allocation evaluates the first four
actual missing cases: fixed water at 0.90, STO-6G+P HBr–NH3 at 1.00, fixed
π-stacked benzene at 2.00, STO-6G+P acetic-acid dimer at 1.00. These are retained
results rather than extra replays. It must produce a passing gate before the
full continuation runs. CPU/GPU hardware is recorded per case.

Continuation requests **one A800 GPU, 32 CPUs, 128 GiB, at most three hours**,
after successful short validation. It resumes the same atomic per-case report,
retaining all passed results and explicit failures. No training job is stopped
or edited. Actual process max RSS and elapsed times are recorded.

The user subsequently approved one serial continuation on 2026-10-26:
**109220**, `afterany:109068`, the same A800 / 32 CPU / 128 GiB, at most three
additional hours. It retains all accepted cases and performs no SCF if the
parent already finished. Plot job 109078 now depends on `afterok:109220`.
The local delivery monitor follows the dynamic submission index, including
the continuation, with an eight-hour finite monitoring window. No other
training or calculation job was cancelled or changed by this arrangement.

Both three-hour calculation allocations subsequently reached their time
limits with 183/204 records passed and no numerical failures. On 2026-10-27
the user explicitly approved a final two-hour continuation for the remaining
21 records (seven cyclopentane-dimer, fourteen water-dimer cases): **109991**,
same single A800 / 32 CPU / 128 GiB, resuming accepted data without replay.
Plot job 109078 now uses `afterok:109991`; the finite local delivery watcher
has been restarted to track that job chain.

## Figures and paper handoff

Generate two independent figure sets, with and without CP, retaining the
same per-system energy scales and source distance factors. Use the existing
square figure style and available scale-1.00 molecular renders, with distinct
line/marker styles for fixed and adaptive MACE and both STO controls. Save
PNG/PDF/SVG, one 12-page PDF per set, exact plotted-point CSV, source/basis
identity and verification reports. Recompute no energies during plotting.

New results and structures will be added to a separate NNAO_paper supplement;
existing historical archives, publication text and active figure scripts are
preserved. Pending, partial and failed records are never labeled complete.
