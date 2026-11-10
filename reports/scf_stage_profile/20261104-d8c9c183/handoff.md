# SCF stage probe handoff

Updated 2026-11-04. Production code is unchanged from d8c9c183. Worktree:
/Users/jiaoyuan/.codex/worktrees/1b2c/GradSCF, branch perf/bse-tddft-grid3-20261101.
Only diagnostic tools and evidence were added. No merge or push.

## Resource state

The original matrix paused after 521 raw configurations, with no worker
interrupted and no orphan logs. Allocation 115855 ran from 17:30:04 to 17:58:35.
The matrix resumed automatically under 115856, with a 44-hour allocation.
Only one NV5090,64 logical CPUs and256GiB were allocated at a time. Scientific
work ran only under Slurm on jy_gpu_2. Four measured subprocesses plus the
12.221-second smoke total1722.554 seconds, within the30-minute probe budget.

Remote root: ~/private/homefile/gradscf-scf-solver-20261102/.
Profile outputs: results/stage-profile-115855/.
Current matrix log: results/resume-matrix-115856.log.
Current summary log: results/summary-watch-115856.log.
Do not launch another controller on the same matrix output.

## Confirmed findings

The p-benzoquinone anion/def2-TZVP/PBE0/grid3 GKS conventional case has312 spatial
AOs and624 spinor orbitals. On the fixed-five-cycle snapshot, warm K takes
16.294s, J2.557s, XC1.001s and diagonalization0.128s. The full energy/Fock kernel
is19.752s. All11 measured kernels completed their target samples, and all12
consistency checks passed.

An exact scheduling experiment gives K9.812s with2048MB extra workspace and a
block_size ceiling of16. Actual widths are p1/q312, compared with default p1/q9.
The maximum value difference is6.66e-16Ha. Compiler temporary accounting grows
from14.0MB to486.0MB. This is a local value-only kernel result, not a production
SCF speedup, converged-snapshot result or AD acceptance.

GKS does not pass its facade budget to the packed contractor, which defaults
to64MB. CPU s4 uses portable(1); native streaming accepts s8 subject to dtype
and workspace checks. The312-AO s4 buffer has2,384,173,584 elements, above2^31,
so J also tiles. The branch starts at304AO, but no adjacent-size experiment
isolated its performance effect. Preserve the large-index correctness guard.

The new clean25-cycle GKS took559.886s, with exactly the same energy as the
previous709s run. New affinity0..7 uses eight physical cores. Historical CPU8
used0..3 and192..195, four physical cores with SMT. Those wall times cannot be
reported as an optimization or equal-resource speed comparison.

The fixed5 diagnostic facade took161.749s; direct solve_scf span101.902s;
inputs34.785s, including ERI27.429s. It is explicitly unconverged. Do not scale
its wall time by25/5 or add separate kernels as a production-loop partition.

The exact backend_compile_and_load function had278 Python calls/5.5535s;
this is not a unique-executable count. synchronize self time122s is waiting
attribution, not proof that synchronization is the algorithmic bottleneck.

The valid Chrome JSON has exactly1M X events,100516 runtime HLO candidates,
and no new events beyond about59.09s, before the SCF scan starts around59.76s.
Suspect an export event cap; do not claim scan op percentages. The155MB xplane
remains remote. The10.64MB Chrome gzip, metadata and pstats are local in raw/
(Gitignored). Source/scope mapping is absent from the Chrome HLO arguments.

UKS true-weigend-RI diagnostic completed25 cycles in85.827s. Inputs8.257s;
auxiliary factors2.392s, including2c/3c1.620s. RI factors occupy417MB and use a
streamed152104-point grid. Its algorithm and representation differ from GKS;
do not report their wall-time ratio as an isolated method comparison.

CPU64 full GKS timed out at600s without an accepted final state. Do not call
that numerical nonconvergence or compute a speed ratio. There was no reported
OOM. No extra profile resource was requested after the bounded run.

## Next implementation boundary

Prioritize consumer budget propagation and bounded CPU tile planning, then
verify complete converged SCF and existing nonsymmetric/batch/complex/JVP/VJP,
geometry and NN regressions. Total memory includes persistent inputs; do not
forward180000MB blindly as extra contraction workspace. Native packed
consumption and thread ownership come next. No new integral backend or physical
approximation is needed for the first experiment.

See report.md, kernel_measurements.csv, stage_measurements.csv,
compile_counts.json and environment.json. cases/ contains exact inputs;
commands.txt provides rerun commands. The original60-system acceptance and
pyridine-cation UKS/GKS energy mismatches remain separate and unresolved.
