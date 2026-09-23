# Degenerate real Ritz bases and explicit restricted SCF multistart

Approved scope: fix CO's degenerate left/right basis robustness, then turn the
N2 native restart workflow into an explicit reusable branch-selection utility.
Continue in feat/eom-cc; no merge or push in this increment.

Root cause: independent eig(A) and eig(A.T) can represent an exact repeated real
root as different real or complex-conjugate bases at roundoff. Taking .real of
each eigenvector can destroy rank. Handle only real clusters unresolved at a
matrix-scaled roundoff threshold. Use left/right null-space bases from one real
SVD of A - lambda_cluster I, form the full cluster overlap, and obtain the dual
basis by a checked solve before truncating to requested roots. Keep isolated
and genuine complex roots on their existing paths. This also avoids inverting
the full eigenvector matrix when an excluded part of the spectrum is defective.

Repair the shared numerical Ritz extraction, used by both dense and Davidson
solves, rather than adding an EOM-specific workaround. Preserve physical
right/left residual checks, conditioning, spectral gaps, and the unchanged
first-order isolated-energy derivative policy. No degeneracy AD or pseudoinverse
is introduced. Projected SVDs remain bounded by the existing search-space limit.

For SCF, expose an opt-in restricted multistart operation. Clone the input facade,
run its baseline settings, reuse orbital_rotation_guesses, rerun each explicit
rotated density, and select the lowest finite converged candidate. Return all
attempt summaries and the selected facade. Do not mutate the source, hide failed
attempts, seed from PySCF, or claim a global-minimum/stability proof. This discrete
host-side selection is separate from differentiation of a selected SCF branch.

Validation: deterministic repeated-root matrices, truncated degenerate clusters,
Jordan/complex/excluded-defect cases, prior matrix-free JAXPR and AD tests; CO
at its originally failing seed/space, N2 native multistart, original eight-system
forward sweep, relevant shared solver/SCF/EOM regressions and independent review.
