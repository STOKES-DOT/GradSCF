# Periodic validation — 2026-09-13

These compact results compare independent GradSCF and PySCF calculations.
The subdirectories retain numerical arrays, figures, source hashes, commands,
software/hardware metadata, and run logs. See each README for tolerances.

- [Silicon and diamond bands](bands/README.md): PBE/GTH-SZV, 2×2×2 SCF k mesh,
  FFT 41³, 61 path points. Maximum band errors are 8.45e-11 and 1.43e-10 eV.
- [Silicon Gamma spectrum](absorption/README.md): full TDDFT, 16 singlet roots,
  velocity gauge including the nonlocal GTH commutator, Gaussian FWHM 0.30 eV.
  Maximum excitation error is 6.96e-11 eV; the maximum spectrum difference
  divided by the reference peak is 2.10e-10.

These establish agreement for the stated finite discretizations. They do not
establish basis/k-mesh convergence or reproduce an experimental bulk spectrum.
The Gamma spectrum is oscillator strength per cell per eV, not a macroscopic
absorption coefficient. Historical runtime metadata records the exact snapshot
used; later cache-validation fixes do not alter the numerical equations.
