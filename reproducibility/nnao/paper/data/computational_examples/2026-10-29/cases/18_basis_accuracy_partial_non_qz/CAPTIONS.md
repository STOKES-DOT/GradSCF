# Standalone figure captions

**DF-MP2.** Database-reference accuracy versus basis size for 15 conventional and nine NNAO bases,
excluding QZ bases. Each point uses the same 30 physical structures to form six ISO34 isomerization
energies and six S22B interaction energies. The vertical axis is their equally weighted mean
absolute error (kcal/mol; logarithmic scale); the horizontal axis is the mean contracted spherical
AO count over all 30 structures. Bubble area is proportional to the mean number of unique spherical
primitive GTO functions, including core functions and angular degeneracy. Calculations use
frozen-core DF-MP2 with one frozen 1s MO per real C/N/O. Interactions are uncorrected E(AB)-E(A)-E(B).
This is a partial 24-basis snapshot of the 27-basis campaign.

**omegaB97M-V.** Same structures, observables, basis set, axes and bubble-area convention as the
DF-MP2 figure, evaluated with omegaB97M-V. Both main and nonlocal-correlation numerical grid levels
are 3; nonlocal correlation is included. Each point compares all 12 relative energies directly to
the frozen database references. No CP/BSSE correction is used. The displayed errors include both
functional and basis contributions and are not a same-method complete-basis-limit error estimate.

**Reference-level statement for both captions.** ISO34 uses the updated W1-F12
composite reference values of GMTKN55 (Goerigk et al., 2017,
[DOI: 10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G), section 2.2.6).
S22B uses the estimated CCSD(T)/CBS reference interaction energies revised by
Marshall, Burns and Sherrill (2011,
[DOI: 10.1063/1.3659142](https://doi.org/10.1063/1.3659142)).
The plotted DF-MP2/DFT interactions are uncorrected E(AB)-E(A)-E(B); no new CP
or BSSE calculation is part of this comparison.
