# NNAO 1000-system period-1--4 restart

Date: 2026-10-13. Keep original L-SZ/200-system job **96510** unchanged.
The previous eight jobs 96633–96640 were archived, then cancelled with explicit
user authorization. Archive:
`GradSCF_basis_grid_20261013/artifacts/retired-200-grid-20261013/` (110 files).

## Frozen input

- 1000 neutral spin=0 structures, at most 30 atoms including hydrogen.
- 800 GMTKN55 v2 + 200 QH9Stable; 181 structures overlap the retained 200-system run.
- First four periods, main-group elements/halogens only. Actual coverage is 25
  elements: H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ge As Se Br Kr.
- Ga has a supported template but no qualified structure in the selected local
  sources. It is not counted as a trained element. Transition metals are not added.
- Sn/Sb/Te/I were removed by the user's period restriction: 58 structures replaced
  by verified eligible structures. Pb/Bi had already been removed.
- `inputs/manifest.json` records all exclusions, replacements, IDs and hashes.
- No held-out split or supervised reaction/Hamiltonian labels.

The old def2-universal-jkfit approximation was checked against native shell-direct
J/K on a core-Hamiltonian density. I/Sb/Te exhibited errors of tens of Hartree;
these are not self-consistent benchmark energies. They are excluded from this
campaign. Cl/Br probe errors were approximately 1–2 mEh. This comparison does not
certify DF accuracy for every molecule or every trained contraction.

## Training protocol

All nine S/M/L x SZP/DZP/TZP configurations start fresh, with seed 0 and independent
Adam states at learning rate 1e-4; target 1000 epochs per configuration. No old
five-element MACE weights or exponent candidate is silently reused.

The same 26 explicit calibration references cover every observed training element.
Each family performs A/B calibration (20 exponent updates maximum; free contraction
300 steps, three starts), selects a valid finite-budget candidate, then performs
fixed-exponent stage C on all 1000 structures. Core policy is `reference`; original
BSE heavy-core records are used without fitting or padding. No ECP is introduced.

Stage C uses mmap S/h/packed DF caches, float64, standard implicit SCF, all-molecule
parameter-direction checks and accepted-state checkpoints. No complete four-index
ERI is stored. Full 1000-molecule compilation residency and epoch timings are not
yet measured; the first full batch is a required gate, not a promised warm speed.

## Three serial GPU lanes

| Lane | Jobs, in order | Device |
| --- | --- | --- |
| SZP | 96694 S → 96695 M → 96696 L | n004 A800, after original 96510 succeeds |
| DZP | 96697 S → 96698 M → 96699 L | one A100_80G |
| TZP | 96700 S → 96701 M → 96702 L | one A100_80G |

All jobs also depend on validation job **96691**. Maximum simultaneous allocation:
three GPUs and 96 CPUs. Every training job requests one GPU, 32 CPUs, 256 GiB,
72 hours. The wall-time request is not a forecast of completion; accepted checkpoints
remain available for explicit continuation if a configuration exceeds its allocation.
No automatic resubmission loop is installed.

## CaO gate repair

The first validation attempt 96691 stopped at S-SZ CaO before any formal training.
Its dependent jobs 96694–96702 were automatically cancelled. Their submission
record is retained as attempt 1; replacement IDs are recorded separately.

At the exact reference geometry, merely raising the iteration limit to 500 did
not recover convergence. A shifted solve with level shift 0.5 and damping 0.2
converged in 55 steps, followed by 3 unshifted/undamped polish steps. Its energy
was -750.9863547092 Eh and minimum internal curvature approximately 0.0893.
The bounded secondary rescue is entered only when the existing primary and
level-shift rescue fail. Final convergence tolerances, raw Aufbau fixed-point
checks and implicit differentiation are unchanged. A coefficient directional
finite-difference regression passes at the original thresholds.

This repair is deployed only to the new 1000-system source directory; original
job 96510 keeps its unchanged source. Replacement dependent jobs remain pending
if a future validation fails, so their dependency can be repaired explicitly;
they never bypass the numerical gate.

## Source and acceptance

New isolated server directory:
`/home/user_yuanjiao/private/homefile/GradSCF_main_group_1000_20261013`.
The original 96510 source and checkpoints are untouched.

Validation 96691 first runs the focused tests, then one real GPU MACE/implicit-SCF
update on all 26 references for every family. Production jobs require all nine
validation records and an exact scientific-source hash match. Queue submission
does not mean those gates or formal training have completed.
