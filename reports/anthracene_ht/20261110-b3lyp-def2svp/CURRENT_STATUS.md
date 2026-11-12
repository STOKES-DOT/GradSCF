# Current status — restarted actual AD run

Verified **2026-11-12 13:19:23+08:00**. Job **122200** runs on jy_gpu_2/n006:
one NV5090,64 logical CPU,256 GiB. Actual anthracene AD began **13:19:20**.
Native geometry AD remains CPU execution. Same B3LYP/Cartesian def2-SVP,
grid3,8 target states,all occupied/virtual orbitals and66 normal modes.

The reverse JIT now receives the saved pullback as dynamic PyTree inputs.
The mechanism test passed (actual lowering warning positive control,
dynamic residual inputs,numerical derivatives and same-pullback compilation
reuse). Native B3LYP/H2 geometry AD passed and was reused with exact hashes.
Two over-specific diagnostic test assumptions were corrected and retained
in the receipts. No numerical tolerance was relaxed.

**0/32 derivatives at launch; no completed AD spectrum.** The job rebuilds
the lost in-memory AD tape; validated ground geometry,Hessian and modes are
reused. Translation and four existing signed FD-direction gates remain
mandatory before spectral generation. The process RSS guard stays220 GiB.

Previous job122119 completed its shared forward in2422.243s, then exceeded
the guard during reverse compilation (112.57GB captured constants;
220.452GiB sampled peak). It stopped11:43:58 and released11:58:59. Those
logs and the vertical result are preserved; no FD production restart occurs.

Remote root:
`/home/user_yuanjiao/private/homefile/gradscf-ad-ht-restart-20261112-1306`.
See [restart manifest](../../anthracene_ad_ht/20261112-9b087fad/restart_122200.json)
and [fixed plan](../../anthracene_ad_ht/20261112-9b087fad/PLAN.md).
