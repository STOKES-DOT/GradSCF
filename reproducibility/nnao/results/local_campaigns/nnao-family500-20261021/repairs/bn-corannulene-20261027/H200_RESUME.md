# L-TZP H200 continuation

The user explicitly requested one H200 on 2026-10-27 and accepted queueing.
Latest lwcluster/0.8.0 manual and live Slurm configuration were checked.
n007 has two typed H200 GPUs and 515109 MiB (about 503 GiB) schedulable host
memory. A 512-GiB request cannot fit. This continuation therefore requests
**one H200, 32 CPUs, 448 GiB host RAM, 72 hours**, with the established
five-minute time-limit requeue signal and atomic checkpoint resumption.

Scientific sources remain frozen at
`c28a091033269e2a50f935050ab481e2480fa5566ef2a45099a6e5a47703179c`.
The same 500 structures, primitive pool, basis size, float64, SCF/implicit
tolerances, batch size 50, seed 0, Adam state, clipping, learning-rate schedule
and early-stop rule are retained. No production kernel or integral is changed.
The accepted checkpoint is update 3, epoch 0, batch cursor 3; it is archived
before submission without rewriting its parameter or optimizer payloads.

On allocation, `train_h200.py` verifies the device type and all frozen source/
input identities, then runs the exact failed BN-Corannulene case in a child
process. That gate compiles and evaluates the full fixed-pool projection,
implicit SCF contraction gradient and MACE parameter backward on H200. It
writes `probe-h200.json` separately from the failed A100 `probe.json`. Only a
passing H200 gate permits training to resume. The child process exits before
the normal training preparation, releasing its executables and device memory.
The gate is reused on later requeues; no extra epoch replay or finite
differences are introduced.

Checkpoint identity contains the hash of seed-0 initialization, although actual
training resumes the saved update-3 vector and Adam moments. The launcher
temporarily wraps the model factory to return the original archived seed-0
vector for that initialization identity. It first requires the newly created
model's initialization to agree within backend-rounding tolerance and checks
the frozen parameter mask. The recorded seed-0 SHA256 is
`4f76dec5b298a0e5617117ed0009e76dec0640cb4b99983d7cebd8f7006416a0`.
This preserves the checkpoint unchanged across GPU architectures; it does not
reset trained parameters, change the network or loosen resume checks.

The clean-process A100 probe also failed the 29.95-GiB runtime allocation:
compiled arguments 5775331360 bytes, temporaries 32162256216 bytes, and a
reported allocator limit 63821905920 bytes. This demonstrates that the
observed failure is not solely accumulated history from earlier molecules.
No claim is made that H200 solves every subsequent large-case memory peak
until the actual gate and training steps pass.

`h200-submitted.json`, `h200-queue-verification.json`, `execution/` and
`probe-h200.json` distinguish submitted/pending/validated/resumed states.
The dimer continuation and automatic paper delivery are independent and
are not changed by this H200 request.

## Observed submission and allocation

Job **110035** was submitted at 2026-10-27 10:38:35 +08:00 and allocated
n007 at approximately 10:40. The driver reports **NVIDIA H200 NVL,
143771 MiB** total device memory (about 140.4 GiB). At 10:42 the execution
record was RUNNING / `h200_exact_case_gate`, resuming update 3; no claim of a
passed H200 gate or a new training update is made at that observation.

At 10:45 the H200 gate had **passed**: BN-Corannulene energy
-797.5908515198535 Eh, 28 SCF cycles, stable=True, fixed-point residual
1.06524e-9, finite contraction and full MACE parameter gradients. Preparation,
compilation and the complete backward took 208.91 s in the clean child process.
JAX reported 37.39 GiB peak active device allocations and 65.0 GiB peak pool
size, distinct metrics; compiled temporary workspace was 31.80 GiB.
The parent then entered resume preparation. At 10:52 it was still running,
with seed-initialization roundoff exactly 0.0 and the accepted checkpoint
still at update 3. A newly accepted training update is not yet certified.
