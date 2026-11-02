# C6Cl6 SCF metric repair (2026-10-26)

The L-TZP job 104598 failed in epoch 1, batch 1, after 33/50 samples and before
any accepted update. The exact initial MACE vector is retained in
`runs/nnao_tzp_l/train/failed-update-000001.npz` on the frozen campaign server.

Recorded diagnostics: energy -2983.735668098415 Eh, AO fixed-point residual
3.5111736179353534e-7, AO commutator norm 1.8797099302831573e-11,
minimum overlap eigenvalue 3.6384739132373226e-7, SCF converged=False after
1200 total rescue cycles. The stability eigensolver residual was
5.8139426351566814e-8; this is a different failure from Ne6/Kr3 eigenpair
resolution. The physical MO gradient still needs independent inspection.

`diagnose.py` compares the original AO solver with an invertible Cholesky
change of coordinates on identical projected S/h/DF factors and exact seed-0
coefficients. It retains every AO and all original SCF tolerances, transforms
the atomic seed consistently, and records AO/orthonormal fixed-point norms,
MO gradient, energy, electron count and density differences. No production
source is modified, and no finite differences or extra training-set passes
are requested.

Diagnostic job **108571**: CPU only, n006, 32 CPUs, 64 GiB, one-hour limit;
submitted after 108566 stopped on an overly strict byte-equality check of
fresh CPU initialization versus saved GPU initialization (maximum difference
3.45e-15). The revised check allows backend rounding; all physical calculations
still use the exact saved GPU vector. Its result is not yet a validated fix. The
healthy L-DZP job 104929 remains unchanged. Any eventual L-TZP formal restart
must retain A100 80G / 512 GiB / 32 CPUs and isolated, hash-recorded sources.

## Confirmed cause and implementation

CPU job 108571 completed the exact-basis comparison. In the original AO
representation SCF exhausted 1200 rescue cycles, with AO fixed-point norm
6.6920e-7 but orthonormal norm 1.1475e-10 and MO gradient 1.1337e-10.
The invertibly whitened solve converged in 32 cycles, with orthonormal norm
3.9333e-11. Energies differed by 3.9e-11 Eh; the orthonormal density difference
was 9.91e-10. These are diagnostic primal results, not a full-gradient gate.

The private NNAO DF-RHF objective now transforms S/h/DF factors using a
Cholesky orthogonalizer and attaches the existing implicit fixed-point rule
to the orthonormal density. Cholesky and integral transformations remain
differentiable. Original overlap eigenvalues and an additional AO residual
are retained for diagnostics. Atomic and explicit AO seeds are transformed
consistently and detached only as algorithmic seeds. All AOs, exponent pools,
Hamiltonian approximations, SCF tolerances and implicit tolerance are retained.

Only `model/nnao/training/objective.py` differs from frozen Kr3 source v2.
The new isolated source hash is
`c28a091033269e2a50f935050ab481e2480fa5566ef2a45099a6e5a47703179c`.
No accepted L-TZP checkpoint exists: restart uses seed 0 and fresh Adam, exactly
as the failed initial run. Its failure snapshot and original metadata are
archived with checksums; no accepted parameter updates are discarded.

## Verification and launch gates

- The new metric regression failed on the old solver and passed after repair.
  It checks value/implicit gradient/HVP under an invertible ill-conditioned AO
  change and against independently unrolled symmetric-orthogonalization SCF.
- Local focused tests: **11 passed, 1 deselected**; no finite differences.
  The preexisting local native library is stale, so a native H2 seed test could
  not run locally. No native sources or binaries were changed for this repair.
- Remote regression must override the root pytest `pythonpath=["src"]`;
  otherwise pytest silently imports the original campaign source. Job 108604
  exposed this test-path issue. Job **108607** uses `-o pythonpath=` and passed
  the metric regression, then runs the real C6Cl6 full-backward gate.
- `validate_fix.py` checks the exact failed seed, 240 AOs, energy agreement,
  stable SCF, existing residual thresholds, and finite coefficient/MACE
  gradients. It writes `validation.json` only on success.
- `train.py` requires that CPU gate and performs one additional exact-case
  gate on the allocated A100, in a subprocess that releases its compilation
  memory before training. A matching successful GPU gate is reused on requeue.
  Neither gate introduces epoch/proposal replays or finite differences.
- Formal training preserves 500 structures, batch 50, micro-batch 1, LR 1e-3
  halved every 100 epochs, 1000-epoch cap, and the existing 10-interval/0.1-mEh
  online-mean stopping rule. The CUDA command buffer remains disabled.

Submission/status evidence belongs in `submitted.json`, `queue-verification.txt`
and `execution/`. A submitted job is not evidence of an A100 gate or a training
update. Healthy L-DZP and completed S/M families are never restarted here.

## Verified outcome and replacement submission

Remote CPU validation **108607 passed**: metric regression plus full real-case
SCF/implicit contraction/MACE backward, 57.0 s for the real-case preparation,
compilation and backward. The real problem converged in 36 SCF cycles without
rescue; energy -2983.7356680984203 Eh, fixed-point norm 1.1564e-12,
orbital residual 8.3580e-10, minimum curvature 0.6609495106,
stability eigen-residual 4.6866e-8. Coefficient and parameter gradient norms
were 2.07020639 and 5.07096129; both complete arrays were finite.

Replacement L-TZP job **108612** was submitted at 2026-10-26 13:28:14 +08:00,
requesting A100 80G / 512 GiB / 32 CPUs / 72 h. It was observed **PENDING
(Priority)** immediately after submission, with no dependency and unknown
start time. The A100 exact-case gate and formal parameter updates remain
unverified until the job starts. L-DZP 104929 was left running on its original
frozen source; only the last L-TZP row of the campaign index was replaced.
