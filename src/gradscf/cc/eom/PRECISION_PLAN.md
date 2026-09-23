# Shared SCF branch and post-HF precision diagnostics

Scope authorized after the stretched/weak-interaction study: improve finite
branch coverage, internal stability and SCF->CC->EOM precision reporting without
duplicating numerical engines or per-method retry loops.

Extend the existing restricted multistart loop with multiple explicit seeds;
run its baseline only once. Record seed and optional stability for each attempt.
Keep single-seed calls compatible and all starts bounded by supplied inputs.
An opt-in stable-candidate filter uses the shared stability analysis; ordinary
SCF, CC and EOM defaults do not change. No automatic global-minimum claim or
hidden tolerance changes are introduced.

Extract existing occupation-based rotation geometry from the orbital objective
factory for reuse. A single internal stability core owns gradient/HVP creation,
public Hermitian solving and residual decisions. RKS and UKS retain their own
existing energy evaluators; packed AO ERIs stay packed. This stage diagnoses real
restricted-internal stability, not RKS->UKS or complex instability. Existing
unrestricted stability/escape behavior remains available through the same core.
Positive partial Ritz curvature is an estimate, not a global certificate.

SCF diagnostics report its existing half-angle gradient norm ||2 F_vo|| and
metric/electron checks from a fresh stored result. EOM precision reporting holds
references to existing SCF/CC/EOM result objects, with caller-selected residual
targets; no replicated CC equations or EOM solve. Failed targets identify the
upstream layer to revisit, not a mathematical bound on excitation-energy error.
Freshness checks reuse existing configuration signatures and array fingerprints.
No measured final deltaE/deltaD is invented when SCF did not retain it.

Acceptance: R/U curvature against dense/independent oracles, compressed-integral
RKS support, unresolved/nonstationary guards, multi-seed F2 branch coverage,
source immutability, stale-state rejection, He2 precision diagnostics, existing
stability and EOM regressions, and independent duplication/correctness review.
