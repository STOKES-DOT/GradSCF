# Initial EOM-CCSD EE/IP/EA increment

Scope selected by the user: implement all three sectors in the first increment.
Start from real molecular closed-shell CCSD: EE singlet and IP/EA doublet.
Use a bounded shared non-Hermitian dense reference solver first; matrix-free
iterative eigenpair solving, open-shell/triplet/SF sectors, transition densities
and oscillator strengths remain later increments. Sector actions are separate
from numerical eigenpair and response ownership.

EE uses the Jacobian of the existing physical CCSD residual in its independent
packed coordinates. IP and EA use spin-adapted charged-sector contractions and
reuse the restricted CC intermediates. Numerical left eigenvectors are duals
in the same packed representation, not the ground-state Lambda amplitudes.

Every sector must expose converged right/left residuals, L.T R=I, spectral gaps
and eigenpair conditioning. Complex roots, unresolved degeneracy and defective
or ill-conditioned eigenpairs keep diagnostics but do not claim real isolated
energy derivatives. Use d omega = l.T (dA) r with l.T r=1 and stopped numerical
eigenvectors. Functional CC->EOM composition retains the ground-amplitude
response. No eigenvector/property AD is promised by this initial energy mode.

PySCF reference conventions are checked at installed version 2.9.0, independent
of online source changes. Compare sector actions before energies, then check
finite differences with CC reconverged. H2/STO-3G additionally permits exact
small-particle-number FCI checks. No production runtime PySCF dependency.

IP eigenvalues are E(N-1)-E(N). EA eigenvalues follow the PySCF attachment
convention E(N+1)-E(N); positive electron affinity is the negative of that value.
Do not discard negative attachment energies. Frozen cores/virtuals follow the
existing CC active-space definitions and remain part of the reference Fock.

Implemented facade: EOMEE/EOMIP/EOMEA(cc_object).run(), plus explicit singlet EE,
IP and EA convenience methods on CCSD. Do not advertise a generic eeccsd method
as covering all spins when only the singlet sector is implemented.

Acceptance: fixed-input PySCF actions/energies, left/right transpose checks,
biorthogonal normalization, real/complex/degenerate/defective solver cases,
frozen spaces and empty sectors, invalid CC/stale-state rejection, JIT/JVP/VJP
energy checks with full CC amplitude response, existing CC/solver regressions,
and native GradSCF examples without CLI construction.
