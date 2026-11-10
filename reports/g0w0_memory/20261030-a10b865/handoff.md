# G0W0 optimization handoff

Baseline: a10b865. Production commits: c79d488 (isolated residue change) and
be50190 (selected preparation, budget, shared screening and independent RI).
Recovered persistent branch: perf/molecular-g0w0-memory-recovery-20261030.
No merge or push. Main's independent iKS update 4d67479 was preserved.

The final remote code matches all 27 accepted source/test hashes. The actual
candidate-native CDLL path and /proc mappings were recorded; library SHA256
is c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b.
Tests ran only on n003 inside allocation 113327: 35 narrow, 5 geometry/compile
and 2 GPU derivative tests passed. Allocation released after verification;
squeue subsequently showed no active job.

Key behavior: projected residue solves and immediate reduction, active CPU
pole skipping, ordered selected MO factors, shared imaginary Cholesky across
RHS/spin frames, optional checked complex GMRES and budgeted CPU/GPU policy.
Independent RI after direct SCF reconstructs F-h-J using the original SCF
Coulomb action. Six-seed QP selection, residuals, Z and branch semantics remain.

Use report.md and before_after.csv for final numbers. Accepted candidate
benzene/GPU files supersede intermediate snapshots. The baseline benzene
sample has cold + ONE complete warm; dispersion is unknown. Process HWM
before/after has different observation windows. The CPU8 pilot was stopped
without QPs, not a timeout or valid speed ratio. No universal scaling claim.

Independent PySCF uses explicit vhf_df=True to match RI bare exchange;
default exact exchange explains the initial 2.631 meV difference. Original
MOLGW source and binary were unchanged; runtime and common basis DATA inputs
were supplied explicitly. Cross-software timing ratios remain unset.

Remaining boundaries: full packed 3c and selected factor caches must fit;
no out-of-core or full RI nuclear/auxiliary-metric AD. NNAO acceptance is a
synthetic four-parameter H2 fixed-geometry/metric fixture, not pretrained
MACE/force-supervised/full GPU training. Periodic/self-consistent GW and BSE
scale limits were not expanded. Refer to API_NOTES.md before new consumers.

Raw logs/XML/input files are in raw/results and the ignored evidence-accepted
archive. analysis.py regenerates machine-readable records and the report
without importing JAX or running calculations. commands.txt gives replay
commands; establish a new authorized Slurm allocation before numerical reruns.
