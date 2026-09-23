# Explicit restricted SCF multistart

`RKS.multistart()` and `scf.run_restricted_multistart()` run a baseline and a
finite set of native orbital-rotation density starts. They select the lowest
**finite, converged candidate** and return all attempt summaries. This is an
opt-in host-side branch-selection workflow. `RKS.run()` is unchanged.

```python
from gradscf import gto, dft, cc

mf = dft.RKS(
    gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", unit="Angstrom"),
    xc="hf", conv_tol=1e-12, conv_tol_density=1e-10,
    conv_tol_grad=1e-9, max_cycle=150,
)
selection = mf.multistart(amplitudes=(0.05, 0.15, 0.4), seed=20260923)
for attempt in selection.attempts:
    print(attempt.amplitude, attempt.energy, attempt.converged, attempt.cycles)
if not selection.converged:
    raise RuntimeError("No finite converged SCF candidate")
mycc = cc.CCSD(selection.selected).run()
```

The input facade is cloned and is not run or mutated. Its existing settings are
preserved for every attempt, including tolerances, damping, integral backend,
basis and molecular geometry. Even an already solved source gets a fresh
baseline calculation, so modified controls cannot silently reuse a stale result.
Only `init_guess` changes for the rotated starts.

The baseline has amplitude 0 and uses the source's configured initial guess.
Additional amplitudes must be finite and positive; `amplitudes=()` requests the
baseline only. `orbital_rotation_guesses` supplies deterministic orthogonal
rotations of the complete baseline MO matrix. Initial densities are
`D = C_rot diag(mo_occ) C_rot.T`, preserving the metric and 0/2 occupations.
No reference-software density or orbitals are used.

`RestrictedMultistartResult` contains:

- `selected`: the selected independent RKS facade, or None if no attempt converged;
- `selected_index`: the index in `attempts`, including the baseline at index 0;
- `attempts`: immutable `RestrictedSCFAttempt` summaries with amplitude, energy
  (Hartree), convergence and iteration count;
- `seed` and the derived boolean `converged`.

Nonconverged/invalid-energy candidates are retained in the summaries but cannot
win selection. Configuration/backend exceptions are propagated rather than
silently treated as ordinary convergence failures. Restart orbitals must be
finite, real and complete, with integer restricted 0/2 occupations.

This does not establish a global minimum or perform a stability analysis.
Selection can switch branches discretely and is not a JIT/AD interface. Use the
existing SCF differentiation APIs on the selected branch separately. The native
N2/STO-3G example demonstrates escape from one observed high-energy branch; it
is not a guarantee for other geometries or strongly correlated systems.
Unrestricted/generalized/fractional-occupation multistart is outside this helper.
HF was validated here; other supported RKS functionals retain their existing
backend/dependency requirements.

See [the native example](../../../examples/cc/nitrogen_multistart.py) and
[the associated EOM validation](../cc/eom/ROBUSTNESS_VALIDATION.md).
