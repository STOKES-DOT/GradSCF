# Spin-channel analysis and bounded negative-curvature following

Continue the existing feat/eom-cc worktree. Add an explicit real spin-breaking
channel to restricted stability, and reuse one bounded restart engine for old
UHF/UKS integral workflows and new facade-level stabilization. No default SCF
model change, independent diagonalizer, energy formula or restart loop.

Spin coordinates use opposite alpha/beta rotations (+theta,-theta). The energy
is the existing unrestricted evaluator with matched configuration and packed
integrals. Check full unrestricted stationarity, not only the identically zero
spin-projected gradient of an unpolarized state. These coordinates have the same
single-spin angle scale as restricted internal rotations; curvature is four times
PySCF's RHF->UHF response action. Verify the factor with an independent oracle.

Expose channel='internal' (default) or 'spin' for restricted analysis. A spin
result supplies two sets of orbitals. Stabilization in that channel explicitly
permits a transition to UKS/UHF; accepted unrestricted states are then checked
in the full real unrestricted tangent space. Complex/generalized instabilities
remain outside this increment. Preserve source objects and stale-state guards.

Retain the resolved negative direction in the common result. The single restart
engine tries finite positive/negative step sizes, accepts only finite converged
lower-energy states, records rejected attempts and termination reasons, and never
relaxes thresholds. Reuse the existing occupation/Cayley routines and SCF runners.
No hidden random retries or dense fallback. Global minima remain uncertified.

Tests: stretched-H2 internal versus spin modes and independent PySCF Hessian;
full stationarity gate; directed N2/F2 descent; native RHF->UHF transition and
source immutability; both signed trials, all-failed trials and budgets; legacy
UHF/UKS stability/escape, packed ERIs, and public API regressions.
