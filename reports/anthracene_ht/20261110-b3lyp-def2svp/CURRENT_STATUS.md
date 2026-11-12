# Completed — eight-state anthracene AD FC/HT spectrum

Verified **2026-11-12 22:28:08+08:00**. Job122200 completed its AD stage at
22:12:35 and spectrum stage at22:12:50; both process receipts report exit0.
The allocation is no longer in squeue. Resources are released.

- All32 finite Cartesian gradients:8 excitation energies and24 signed dipole components.
- All66 modes retained; four independent FD directions and translation gates passed.
- Final eight-state FC/HT/interference spectra,common-axis area checks and AD provenance passed.
- All four actual PNG figures were opened locally for visual review.

AD process wall:31993.843s (8h53min14s),including startup,compile,shared
linearization,32 pullbacks and validation. Process-group RSS sampled peak:
176860991488 bytes (164.7146GiB,0.2s sampling),below220GiB guard.
Spectrum process wall:6.301s. Native geometry AD ran on CPU; the allocated
5090 is not evidence of GPU native geometry execution.

Local outputs: [total spectrum](ad_eight_states/anthracene_fc_ht.png),
[state spectra](ad_eight_states/anthracene_fc_ht_states.png),
[CSV data](ad_eight_states/spectrum.csv),[band table](ad_eight_states/bands.csv).
Final evidence: [summary](../../anthracene_ad_ht/20261112-9b087fad/FINAL_RESULTS.md).

Scope:0K independent equal-curvature VG surfaces and linear HT; no
Duschinsky,excited-state Hessians,thermal populations or coupled electronic-
state vibronic Hamiltonian. No empirical alignment or external experimental
agreement is claimed. Old FD data supplied validation only.
