# GW-family handoff

Production source is committed on `perf/gw-family-memory-20261030` at
`28475e6`, after `ca87549` (qsGW) and `12d2ad7` (Matsubara). Worktree:
`/Users/jiaoyuan/Documents/GitHub/GradSCF/artifacts/side/g0w0-memory-recovery-20261030`.
The baseline is `ea646bf`; earlier a10b865 results are not this comparison.
No merge or push. Original release/v1.0.0 remains at 4d67479 with empty tracked
diffs and its original untracked path list.

Restricted evGW prepares ordered MO vertices/exchange once, caches only
physically fixed W0, and shares the stable numerical G0W0 executable. qsGW
streams mode-A row/column products, factors imaginary screening once per
frequency, and shares retarded zero screening across own poles. Native/packed
conventional G0W0/evGW use actual OV Coulomb blocks, shared checked transition
solves and the existing QP machinery; they never create AO-pair spectral
factors. scGW keeps its equations/tails but uses per-frequency LU work and the
existing compiled canonical charge search in both forward paths.

CH2O/def2-SVP CPU32 (16 physical cores), full G/W and six targets: warm E2E
evGW0 8.6702→2.2616 s, evGW 13.3605→2.0301 s; all six QPs agree below
1e-15 Ha. Three warm samples per engine/method. Fixed-input qsGW 81.94→23.25 ms;
XLA temporary estimate 4.894→0.363 MB. This is a kernel result, not E2E qsGW.
Its compile time increased 1.125→2.244 s. Charge warm improved 5.843→1.673 ms,
but first execution increased 0.680→2.618 s. Screening warm ranges overlap;
its useful result is 1.090→0.099 MB XLA temporary workspace. Whole-process RSS
must not be confused with compiler temporary estimates.

Phenanthrene is PubChem CID 995, fixed computed conformer. def2-SVP:
260 AOs, 974 auxiliary functions, 47 occupied/213 virtual, all G/W, six QPs.
Candidate CPU cold/warm E2E 182.76/175.06 s; CPU RI/SCF→V100 GPU cold/warm
32.78/20.70 s. The GPU warm GW part is 8.93 s. Single warm sample only.
CPU/GPU QPs agree within 1.56e-14 Ha. def2-TZVP GPU capacity: 564 AOs,
47 occupied/517 virtual, same declared weigend fit, first E2E 65.72 s;
CPU preparation 42.06 s, transfer 0.74 s, GPU GW 22.87 s. This is a capacity
point, not a basis-convergence or experimental-accuracy claim.

Original affected gates: 129 passes. Final affected sequence: 74 passes;
then 20 conventional and 8 facade checks after their last patches. These
executions overlap. Native H2 full geometry JVP/VJP and reconverged two-step
FD passed. Synthetic H2 radial RI→implicit HF→QP loss/parameter FD/Adam passed;
it is not pretrained MACE, force-loss, or full RI geometry validation. Separate
complete-output checks gave qsGW max error 1.39e-17 and zero screen/charge
differences. All final acceptance/performance/controller exits are zero.

Limits: true conventional qsGW/scGW/U-GW and conventional BSE export are
explicitly unsupported. Conventional V_OV cache remains quadratic in OV
space; no cache-free or disk backend. Full scGW Pi/W/time caches and its
complete common budget remain open in PLAN.md. Existing factor/BSE entry
points retain actual consumers. GPU measurements cover fixed-geometry G0W0,
not general GPU training. No fresh PySCF/MOLGW ratios were produced this round.

Keep the initial combined exit 139 log: after 30 tests the JAX compiler crashed.
The pre-fix water case passed in isolation; tracing was separately reproduced
and fixed. Do not claim the native crash's root cause is proven. CPU model and
linked BLAS version were not captured; actual affinity/core topology and thread
limits are recorded instead. Every runtime GW source hash matches the local
production commit. Native library SHA256 is c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b.

Slurm allocation 113525 (one V100-SXM2-32GB, 32 logical CPUs, 256 GiB) is released;
squeue returned no job. Remote source/results remain under
`~/private/homefile/gradscf-gw-family-memory-20261030/`. Raw logs/NPZ/XML are
preserved locally in ignored `raw/`; normalized JSONL/CSV and source hashes are
tracked. See commands.txt and scripts for reruns inside a fresh allocation.
