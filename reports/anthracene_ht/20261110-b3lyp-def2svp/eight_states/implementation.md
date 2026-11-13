# Eight-state extension

User expanded the target on2026-11-11. All8 lowest singlets receive VG/linearHT,
including dark states, with66common validated modes.12candidate roots provide
upper guards; Hungarian signed transition-density matching is one-to-one.

No ground geometry, Hessian or mode recomputation. OldS1points lack otherstate
signeddipoles, so expanded points require newsharedSCF/TDDFT solves. OldS1
observables are phase-aligned crosschecks, not invented eight-state inputs.
Every expanded point retains rawC/X/Y/E/dipoles for allcandidates.

TDD receipts: tracking14REDfail ->14GREENpass, previoustracking4pass;
spectrum3REDfail+1pass ->4GREENpass; single-statefactorized14pass unchanged.
Combined36pass6.61s, independentreadonlyreviewfoundnoblockingdefect.

Actualbaseline12roots converged. First expanded point completed524.36s; all8
matched withminoverlap0.99999894 andoldS1signeddipole difference4.51e-9ebohr.
Thus the originalsingle-state4–5h estimate no longerapplies. Newpoint rates
are recorded rather thanassuming eighttimescost orunchangedcost.

All spectra useoneabsolutephotonaxis, independentstateintensitysum andsigned
withinstateFC×HTinterference. Perstateandtotalopticalarea checks1e-5; no
perbandnormalization. Finalplotsarependingallprimarypoints andhalfstepchecks
(unionofperstate topHR/topHT modes, atmost16modes/32extra points).

Currentoriginalallocation119234 remainsoneNV5090/64logicalCPU/256GiB. Ifits
mode-process timeguard expires beforecompletion, aconditional sequential24h
allocation canresume unchangedsignedcheckpoints onlyaftertheparentends.
It isnotyet submitted, neveraddsconcurrentcards andneverblindlyretries
scientificerrors. FinalPHYSICALspectrum isNOTcomplete.
