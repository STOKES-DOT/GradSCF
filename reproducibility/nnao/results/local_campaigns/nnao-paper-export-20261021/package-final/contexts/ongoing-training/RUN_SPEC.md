# Eight-family NNAO training on the same 500 structures

The user approved the existing L-SZP run's training settings and a maximum
of three GPUs, grouped by valence zeta. Existing L-SZP is retained.

| Lane | Serial order | Resource per job |
| --- | --- | --- |
| SZP | S-SZP, M-SZP | A800 x1 on n004, 32 CPU, 448 GiB, 72 h |
| DZP | S-DZP, M-DZP, L-DZP | A100_80G x1, 32 CPU, 512 GiB, 72 h |
| TZP | S-TZP, M-TZP, L-TZP | A100_80G x1, 32 CPU, 512 GiB, 72 h |

All three first jobs wait for one prerequisite preparation job; successors
have `afterany` dependencies within their lane. An unsuccessful family is
recorded explicitly and does not prevent an independent family from
starting. The dependency chains allow at most three simultaneous GPUs.
The prerequisite requests one A800 on n005/32 CPU/512 GiB for at most 4 h and
finishes before any family job starts. No other training is cancelled.
These are resource limits, not a guarantee of finishing 1000 epochs in one
allocation. Each family preserves Adam and per-batch checkpoints for resume.

## Matched training definition

Inputs are exact copies of `mace500-20261014/inputs/references.json`,
`model-config.json`, and `pilot.json`; checksums are recorded in
`campaign.json`. The same 500 structures cover 25 elements, are neutral,
spin=0, and have at most 30 atoms. No validation split or sample removal.
Only `basis_family` changes in the data/model definition for each run.

The MACE backbone is 256x0e + 256x1o, 2 interactions, max_ell=3,
correlation=3, 8 Bessel functions, radial MLP 64/64/64, readout 16x0e,
cutoff 5 A, SiLU. Seed 0 and fresh Adam for each family; only the required
coefficient head dimensions change with the basis. Float64 DF-RHF,
def2-universal-jkfit, existing implicit SCF and convergence rules.

Batch size 50, one molecule at a time, 10 updates per complete epoch.
Adam starts at 1e-3, halves every 100 epochs; beta1=.9, beta2=.999,
epsilon=1e-8, no weight decay, global gradient norm clipped to 1.
Maximum 1000 epochs. One measured forward/backward per sample; no startup
replay, proposal evaluation, full-epoch scoring, or finite differences.
SCF/gradient failures stop that run without advancing its accepted state.

## Early stopping

Let L_e be the logged online mean SCF energy of all 500 structures over
epoch e. After 11 complete consecutive epoch means are available, stop if

`sum(abs(L_i - L_(i-1)), i=e-9,...,e) < 1e-4 Eh`.

This is 0.1 mEh in mean energy, not a threshold on the 500-system sum.
Opposite changes do not cancel. Check at each epoch boundary; incomplete
epochs do not qualify. History and the stopping policy survive restart.
The online mean contains ten parameter versions; it is not a full-set
measurement at the final saved vector and does not identify a best model.

## Fixed exponent pools and cache

S reuses the accepted 25-element `nnao_dzp_s` pool from the previous
reference-core campaign. L reuses the actual 500-system L-SZP pool.
Both retain BSE 6-31G core identity
`8e4749c35ca02ea84bd798b87c6c14cd68f43fcefe56f60b4e1b22d3c9dfacda`.
M has no accepted complete pool; prepare it with the same 26 reference
structures, 20 exponent steps and 3 x300 free-DZ-contraction steps per
candidate. Record finite-budget statuses; skip differences explicitly.
Each S/M/L pool is then shared across SZP/DZP/TZP unchanged. Only MACE
coefficients are trained; stage D exponent fine tuning is not launched.

The prerequisite checks each candidate's core, primitive order and absolute
exponents against all 26 reference layouts, then runs exactly one H2
forward/backward update with the full-size network for each new family.
Each smoke run is a separate process. Formal runs start from seed 0,
with fresh Adam, independently of smoke checkpoints.

S/h/packed-DF caches use mmap and are shared by primitive tier across zeta.
Cache v2 identifies the physical pool independently of coefficient-head
slots; a file lock prevents concurrent creation of the same arrays.
The actual contracted AO count remains per-consumer metadata. No full ERI.
The new remote directory is a source export of the current main workspace,
including the UKS fix; old source exports/checkpoints remain unchanged.

At 512 GiB per A100 job, two A100 jobs cannot fit concurrently in n001's
roughly 1007 GiB schedulable memory. SZP can now run independently on n004's
A800; actual concurrency remains subject to node capacity and the three-GPU cap.
The original 500-system L-SZ hot assessment recorded a sampled RSS peak of
337208606720 bytes (about 314 GiB). SZP uses 448 GiB on n004, whose total is
about 503 GiB; this margin is not a measured SZP peak or guarantee for all runs.
A100 requests retain 512 GiB. Both GPU models are permitted for this campaign.

The first preparation job 102758 failed before compilation because the manual's
GCC12 enable path did not exist. Its replacement 102860 uses `/usr/bin/gcc`
and `/usr/bin/g++`, prints the actual version and builds in the artifact directory.
Preparation uses n005's four-hour allocation, within its enforced eight-hour cap.
The 72-hour A800 SZP requests on n004 passed Slurm's test-only validation.
Changing the GPU type of pending jobs in place was rejected, so only the
unstarted affected jobs were replaced and all dependencies were reverified.
No checkpoint or numerical result was removed. See `submitted.json` and
`scheduler-after-a800.json` for current job IDs and controller state.
