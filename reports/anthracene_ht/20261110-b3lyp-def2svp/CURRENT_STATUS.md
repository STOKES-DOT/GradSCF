# Current active calculation

User expanded HT toall8lowest singlets on2026-11-11. Do notinterpret theold
S1raw/mode_response_status.json asactiveprogress.

Activejob119234,step119234.541, oneNV5090+64logicalCPU+256GiB, n006.
Newoutput: eight_states/raw/. Newlog: results/eight-mode-response.log.
Sourcebranch examples/anthracene-ht-20261110, worktreebd0a.

Groundstructure/all72Hessian columns/66positive modes are unchangedvalidated
inputs, notrerun. Eighttargets are jointlymatched to12candidates (4guards).
Darkstates retain signed transition-density phases.

Verified22:13:44+08: baseline12roots converged andgap/phasechecks passed.
Firsteight-statepoint completed524.36s, minoverlap0.99999894; oldS1signed
dipole error4.51e-9ebohr. Remainingpoints/stepchecks/spectrum NOTcomplete.
Combinedtracking/legacy/factorized/multibandplottests36passed6.61s.

Eight-state mode data:(8,66)gapgradient,(8,3,66)dipoleslopes. PerstateFC/HT/
interference/total spectra shareonephysicalphotonaxis, intensities summedincoherently.
No separatebandnormalization, Duschinsky, finiteT orcoupled-electronicLVC model.

Currentallocationexpires2026-11-12about15:00+08. Modeguard60000s endsbefore
that. Ifmodeguard returns timeout124only, queue27 can submit asame-profile
24hsequentialcontinuation dependentafterany on119234; no secondcardcanrun
concurrently. Theconditionaljob hasNOTyetbeensubmitted. Numericfailuresare
notblindlyretried. Ifparentended,inspect results/eight-continuation-job.txt
foractualnextjob, then results/eight-mode-resume.log. NoPySCFresumption.

Remote root:/home/user_yuanjiao/private/homefile/gradscf-anthracene-ht-20261110
Local snapshotsarenot automatically refreshed. No merge/push performed.
