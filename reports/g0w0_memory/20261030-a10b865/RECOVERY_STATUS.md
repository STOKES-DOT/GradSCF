# Recovery and remaining acceptance

**Resolved on 2026-10-30.** SSH was restored. The 27-file comparison passed
with only the intended fixture difference; the corrected narrow gate passed
35/35, geometry/compile 5/5 and GPU array/HVP 2/2. Fresh accepted benchmarks
and complete raw records were retrieved. Allocation 113327 was released and
absent from the subsequent squeue query. `manifest.json` and `report.md`
contain final status. The historical checkpoint below is retained for audit.

The original `/private/tmp/gradscf-g0w0-memory-20261030` directory disappeared
between turns. No cause is inferred. Its committed baseline (`a799f0e`) and
projected-residue change (`c79d488`) remain in Git.

The remaining source/tests/harness have been restored by chronological replay
of the actual file-write tool calls in this task's retained session log:
41 successful patches and 13 Python text rewrites, including the 169-line
native NNAO/SCF/QP test from its dedicated agent. The one originally failed
multi-file documentation patch failed identically during replay and was skipped;
its successful replacement was replayed. No equations were reimplemented from
a prose summary. `recovery.json` records the restored hashes.

All restored Python sources parse, and `git diff --check` passes. No local
numerical tests have been run. Byte comparison against the remote candidate
and final numerical acceptance remain pending SSH restoration.

## Last verification boundary

Earlier remote results: 35 narrow tests passed, plus independently recorded
periodic/resolvent, UGW/AD, geometry/compile and NNAO checks. A later final-source
run displayed one failure. The output position and static array accounting
identify a now-insufficient budget in the weighted-factor test fixture:
22,106,112 resident bytes and at least 33,685,504 screening bytes cannot fit
the old usable 48,000,000 bytes (60 MB minus reserve). The fixture now requests
100 MB and asserts the corresponding 80 MB usable bound. Numerical tolerances
and production equations are unchanged. This correction has NOT been rerun;
retrieve the exact remote traceback and verify it before closing acceptance.

## Remote state

Existing allocation: 113327, n003, one V100, 64 logical CPUs, 256 GiB;
its current status is unknown after connection loss. Do not assume it remains
valid and do not submit another allocation until checking it.

Both resumed `ssh jy_gpu_2` probes timed out at 10.105.17.37:59922 before
authentication. The user is checking the existing connection route. No new
calculation or resource request was submitted during recovery. Resource release
cannot be claimed until the original allocation has been queried/released.

## Resume sequence

1. Query job 113327 and inspect the remote final-source-narrow failure/exit.
2. Download `candidate_source_hashes.json`, `results/` and completed source
   snapshots from `~/private/homefile/gradscf-g0w0-memory-20261030/`; compare
   restored bytes before syncing the fixture correction.
3. Run the corrected memory fixture, final narrow checks, geometry/compile
   checks, and the final candidate benzene repeat if it did not execute.
4. Restore full raw results and finalize the report/CSV/plan status. Release
   only this project's allocation when verification ends. No automatic merge
   or push is authorized by this implementation task.

The original main worktree is separate and now contains the independently
added iKS commit 4d67479. Its tracked files are clean; recovery preserves its
HEAD, index and unrelated untracked work. The GW baseline remains a10b865.
